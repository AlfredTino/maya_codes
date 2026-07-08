import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from utils.data_utils import (TabularData, TaskType, get_num_and_cat_feats,
                              is_integer_tensor)
from utils.ddp_utils import get_rank, get_world_size, print_ddp, save_ddp
from utils.metric_utils import get_metrics

from models._base import BaseTabAnalyzer
from models.modeling.modeling_maya import MayaModel


def make_random_batches(
    train_size: int, batch_size: int, device=None
) :
    permutation = torch.randperm(train_size, device=device)
    batches = permutation.split(batch_size)
    # Below, we check that we do not face this issue:
    # https://github.com/pytorch/vision/issues/3816
    # This is still noticeably faster than running randperm on CPU.
    # UPDATE: after thousands of experiments, we faced the issue zero times,
    # so maybe we should remove the assert.
    assert torch.equal(
        torch.arange(train_size, device=device), permutation.sort().values
    )
    return batches  # type: ignore[code]

class Maya(BaseTabAnalyzer):
    def __init__(self, args, task_type):
        super().__init__(args, task_type)

    def config_convert(self, model_config):
        if model_config.intermediate_size < 10:
            model_config.intermediate_size = int(model_config.hidden_size * model_config.intermediate_size)
            if model_config.intermediate_size > 256:
                model_config.intermediate_size = 256
        if model_config.decoder_configs["decoder_layer_configs"]["decoder_intermediate_size"] < 10:
            model_config.decoder_configs["decoder_layer_configs"]["decoder_intermediate_size"] = int(
                model_config.hidden_size * 
                model_config.decoder_configs["decoder_layer_configs"]["decoder_intermediate_size"]
            )
            if model_config.decoder_configs["decoder_layer_configs"]["decoder_intermediate_size"] > 256:
                model_config.decoder_configs["decoder_layer_configs"]["decoder_intermediate_size"] = 256
        return model_config

    def get_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.model_config["model"]
        model_config = SimpleNamespace(**model_config)
        model_config = self.config_convert(model_config)
        task_type = "regression" if self.task_type == TaskType.REGRESSION else "classification"
        self.model_config = model_config
        try:
            self.model = MayaModel(
                d_numerical=self.d_in,
                categories=self.categories,
                hidden_size=model_config.hidden_size,
                num_heads=model_config.num_heads,
                intermediate_size=model_config.intermediate_size,
                num_layers=model_config.num_layers,
                label_nums=self.d_out,
                dropout_ratio=model_config.dropout_ratio,
                add_act=model_config.add_act,
                if_bias=model_config.if_bias,
                skip_first_norm=model_config.skip_first_norm,
                task_type=task_type,
                last_mlp_skip=model_config.last_mlp_skip,
                act_type=model_config.act_type,
                using_attn_norm=model_config.using_attn_norm,
                num_branch=model_config.num_branch,
                mlp_using_legacy=model_config.mlp_using_legacy,
                using_encoder_decoder_arch=model_config.using_encoder_decoder_arch,
                decoder_configs=model_config.decoder_configs
            )
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()

    def fit(self, data, train=True, info=None, config=None):
        if self.args.distribute:
            self.args.num_tasks = get_world_size()
            self.args.local_rank = get_rank()
            if self.args.local_rank >=8:
                self.args.local_rank = self.args.local_rank - 8
            self.args.device = torch.device("cuda", self.args.local_rank)

        if isinstance(data, tuple) and info is not None:
            self.tabular_data = TabularData(data[0], data[1], data[2], info, None)
        else:
            self.tabular_data = data
        
        if config is not None:
            self.reset_stats_withconfig(config)
    
        self.args.train_dtype = getattr(torch, self.args.train_dtype) if type(self.args.train_dtype) is str else self.args.train_dtype
        self.data_preprocessing(is_train=True)
        self.get_model()
        self.model.to(self.args.train_dtype)
        self.model.to(self.args.device)
        if self.args.distribute:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=False
            )
        params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            params, 
            lr=self.args.model_config["training"]["lr"], 
            weight_decay=self.args.model_config["training"]["weight_decay"]
        )
        total_steps = self.args.max_epoch * len(self.train_dataloader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, total_steps, eta_min=5e-6
        )
        if not train:
            return

        time_cost = 0
        for epoch in range(self.args.max_epoch):
            start = time.time()                
            self.train_epoch(epoch)
            self.validate(epoch)
            cost = time.time() - start
            time_cost += cost
            # print_ddp(f"Epoch: {epoch}, Time cost: {cost}")
            print_ddp(f"dataset: {self.args.dataset_name}, Epoch: {epoch}, Time cost: {cost}")
            if not self.continue_training:
                break
        
        save_ddp(self.model, os.path.join(self.args.save_path, f"epoch-last-{self.args.seed}.pth"))
        return time_cost

    def train_epoch(self, epoch):
        self.model.train()
        mean_loss = 0
        for i, (X, y) in enumerate(self.train_dataloader, 1):
            self.train_step = self.train_step + 1
            X_num, X_cat = get_num_and_cat_feats(X, self.N, self.C)
            self.optimizer.zero_grad()
            pred = self.model(X_num, X_cat, y)
            loss = self.criterion(pred, y)

            loss.sum().backward()
            self.optimizer.step()
            mean_loss += loss.mean().item()
            
            if (i-1) % 50 == 0 or i == len(self.train_dataloader):
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch}, train {i}/{len(self.train_dataloader)}, loss={loss.item():.4f} lr={lr:.4g}")
            del loss

        mean_loss = mean_loss / len(self.train_dataloader)
        self.train_states.train_loss.append(mean_loss)
    
    def prepare_enc_output(self, x, y, candidate_x, candidate_y, is_train, sample_rate=1):
        X_num, X_cat = x
        candidate_x_num, candidate_x_cat = candidate_x
        if is_train:
            # select candidate
            data_size = candidate_y.shape[0]
            retrival_size = int(data_size * sample_rate)
            sample_idx = torch.randperm(data_size)[:retrival_size]
            candidate_x_num = candidate_x_num[sample_idx] if candidate_x_num is not None else None
            candidate_x_cat = candidate_x_cat[sample_idx] if candidate_x_cat is not None else None
            candidate_y = candidate_y[sample_idx]

            # get encoded candidate_x
            enc_candidate_x = self.model(candidate_x_num, candidate_x_cat, candidate_y)

            # get encoded mini-batch
            enc_x = self.model(X_num, X_cat, y)

            return enc_x, enc_candidate_x, candidate_y
        else:
            # get encoded candidate_x
            enc_candidate_x = self.model(candidate_x_num, candidate_x_cat)

            # get encoded mini-batch
            enc_x = self.model(X_num, X_cat)

            return enc_x, enc_candidate_x

    def validate(self, epoch):
        print(f"best epoch {self.train_states.best_epoch}, best val res={self.train_states.best_result:.4f}")
    
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_dataloader)):
                X_num, X_cat = get_num_and_cat_feats(X, self.N, self.C)
                if self.model_config.using_encoder_decoder_arch:
                    candidate_x_num = self.N["train"].to(self.args.train_dtype) if self.N is not None else None
                    if self.C is not None:
                        candidate_x_cat = self.C["train"] if is_integer_tensor(self.C["train"]) else self.C["train"].to(self.args.train_dtype)
                    else:
                        candidate_x_cat = None
                    candidate_y = self.y["train"] if is_integer_tensor(self.y["train"]) else self.y["train"].to(self.args.train_dtype)

                    enc_x = self.model(X_num, X_cat)
                    enc_x_ref = self.model(candidate_x_num, candidate_x_cat)

                    x = enc_x
                    x_ref = enc_x_ref
                    decoder_layers = self.model.module.decoder if dist.is_initialized() else self.model.decoder
                    gen_label_layer = self.model.module.gen_label if dist.is_initialized() else self.model.gen_label
                    for decoder_layer in decoder_layers:
                        decoded_out = decoder_layer(
                            x, x_ref, candidate_y, False
                        )
                        x = decoded_out

                    pred = gen_label_layer(decoded_out)
                    pred = pred.squeeze()
                else:
                    pred = self.model(X_num, X_cat)

                test_logit.append(pred)
                test_label.append(y)
            test_logit = torch.cat(test_logit, 0)
            test_label = torch.cat(test_label, 0)
        if dist.is_initialized():
            dist.barrier()
            gathered_preds = [torch.zeros_like(test_logit) for _ in range(dist.get_world_size())]
            gathered_label = [torch.zeros_like(test_label) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_preds, test_logit)
            dist.all_gather(gathered_label, test_label)
            test_logits = torch.cat(gathered_preds)
            test_labels = torch.cat(gathered_label)
            dist.barrier()
        else: 
            test_logits = test_logit
            test_labels = test_label
        
        test_loss = self.criterion(test_logits, test_labels).item()  
        test_res, _ = get_metrics(
            preds=test_logits, 
            gts=test_labels, 
            y_info=self.data_preprocess_y.info,
            task_type=self.tabular_data.info["task_type"]
        ) 

        if self.task_type == TaskType.REGRESSION:
            task_type = "regression"
            measure = np.less_equal
        else:
            task_type = "classification"
            measure = np.greater_equal

        print(f"epoch {epoch}, val, loss={test_loss:.4f} {task_type} result={test_res:.4f}")
        if measure(test_res, self.train_states.best_result) or epoch == 0:
            self.train_states.best_result = test_res
            self.train_states.best_epoch = epoch
            save_ddp(self.model, os.path.join(self.args.save_path, f"best-val-{self.args.seed}.pth"))
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 30:
                self.continue_training = False
                print(f"***************finish training with best val res={self.train_states.best_result:.4f}***************")
        if self.args.distribute:
            if dist.get_rank() == 0:
                torch.save(self.train_states, os.path.join(self.args.save_path, "train_states"))   
        else:
            torch.save(self.train_states, os.path.join(self.args.save_path, "train_states")) 

    def predict(self, data):
        N, C, y = data
        if self.args.distribute:
            dist.barrier()
            self.model.module.load_state_dict(
                torch.load(
                    os.path.join(self.args.save_path, f"best-val-{self.args.seed}.pth"), 
                    map_location="cpu"
                )["params"]
            )
        else:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.args.save_path, f"best-val-{self.args.seed}.pth")
                )["params"]
            )
        print(f"best epoch {self.train_states.best_epoch}, best val res={self.train_states.best_result:.4f}")

        self.model.eval()
        self.data_preprocessing(
            is_train=False, N=N, C=C, y=y
        )

        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.test_dataloader)):
                X_num, X_cat = get_num_and_cat_feats(X, self.N, self.C)

                if self.model_config.using_encoder_decoder_arch:
                    candidate_x_num = self.N["train"].to(self.args.train_dtype) if self.N is not None else None
                    if self.C is not None:
                        candidate_x_cat = self.C["train"] if is_integer_tensor(self.C["train"]) else self.C["train"].to(self.args.train_dtype)
                    else:
                        candidate_x_cat = None
                    candidate_y = self.y["train"] if is_integer_tensor(self.y["train"]) else self.y["train"].to(self.args.train_dtype)

                    enc_x = self.model(X_num, X_cat)
                    enc_x_ref = self.model(candidate_x_num, candidate_x_cat)

                    x = enc_x
                    x_ref = enc_x_ref
                    decoder_layers = self.model.module.decoder if dist.is_initialized() else self.model.decoder
                    gen_label_layer = self.model.module.gen_label if dist.is_initialized() else self.model.gen_label 
                    for decoder_layer in decoder_layers:
                        decoded_out = decoder_layer(
                        x, x_ref, candidate_y, False
                        )
                        x = decoded_out

                    pred = gen_label_layer(decoded_out)
                    pred = pred.squeeze()
                else:                                                                        
                    pred = self.model(X_num, X_cat)

                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        if self.args.distribute:
            dist.barrier()
            print(f"test logit is {test_logit[0].shape}")
            gathered_preds = [torch.zeros_like(test_logit) for _ in range(dist.get_world_size())]
            gathered_label = [torch.zeros_like(test_label) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_preds, test_logit)
            dist.all_gather(gathered_label, test_label)
            test_logits = torch.cat(gathered_preds)
            test_labels = torch.cat(gathered_label)
            test_loss = self.criterion(test_logits, test_labels).item() 
            test_res, metric_name = get_metrics(
                preds=test_logits, 
                gts=test_labels, 
                y_info=self.data_preprocess_y.info, 
                task_type=self.tabular_data.info["task_type"]
            )   
            dist.barrier()
        else:
            test_loss = self.criterion(test_logit, test_label).item()     
            test_res, metric_name = get_metrics(
                preds=test_logit, 
                gts=test_label, 
                y_info=self.data_preprocess_y.info, 
                task_type=self.tabular_data.info["task_type"]
            )

        print(f"Test: loss={test_loss:.4f}")
        print(f"[{metric_name}]={test_res:.4f}")

        return test_loss, test_res, metric_name, test_logit
