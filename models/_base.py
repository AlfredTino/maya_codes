import dataclasses as dc
import os
import time
import typing as ty
from dataclasses import field

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from utils.common_utils import Averager, get_device, set_seeds
from utils.data_utils import (TabularData, TaskType, get_categories,
                              get_data_by_split, get_dataloader,
                              get_num_and_cat_feats)
from utils.ddp_utils import get_rank, get_world_size, is_main_process, save_ddp
from utils.metric_utils import get_metrics


@dc.dataclass
class TrainStates:
    args: ty.Any 
    best_epoch: int
    best_result: float
    train_loss: ty.List = field(default_factory=list)

class BaseTabAnalyzer:
    def __init__(self, args, task_type):
        self.args = args
        if is_main_process:
            print(args.model_config)  
        self.task_type = task_type 
        self.tabular_data = None

        self.train_step = 0
        self.val_count = 0
        self.continue_training = True

        self.train_states = TrainStates(
            args=vars(args),
            best_epoch=0,
            best_result=1e10 if self.task_type == TaskType.REGRESSION else 0,
            train_loss=[]
        )

        self.args.device = get_device()
    
    def get_model(self):
        raise NotImplementedError

    def reset_stats_withconfig(self, config):
        set_seeds(self.args.seed)
        self.train_step = 0
        self.val_count = 0
        self.continue_training = True
        self.config = self.args.model_config = config
        
        self.train_states = TrainStates(
            args=vars(self.args),
            best_epoch=0,
            best_result=1e10 if self.task_type == TaskType.REGRESSION else 0,
            train_loss=[]
        )

    def data_preprocessing(self, is_train=True, N=None, C=None, y=None):
        if is_train:
            self.y, self.data_preprocess_y = self.tabular_data.build_y()
            (self.N, self.C), self.data_preprocess_x = self.tabular_data.build_X(
                normalization=self.args.normalization,
                cat_policy=self.args.cat_policy,
                seed=self.args.seed,
                y_train=self.y["train"]
            ) 
            
            if self.tabular_data.is_regression:
                self.d_out = 1
            else:
                self.d_out = len(np.unique(self.y["train"]))
            self.d_in = 0 if self.N is None else self.N["train"].shape[1]
            self.categories = get_categories(self.C)
            self.N, self.C, self.y, self.train_dataloader, self.val_dataloader, self.criterion = get_dataloader(
                self.args, self.tabular_data.is_regression, (self.N, self.C), self.y, self.data_preprocess_y.info, is_train=True
            )
        else:
            y_test, _ = self.tabular_data.build_y(data_preprocess_y=self.data_preprocess_y, y_test=y)
            (N_test, C_test), _ = self.tabular_data.build_X(
                normalization=self.args.normalization,
                cat_policy=self.args.cat_policy,
                seed=self.args.seed,
                data_preprocess_x = self.data_preprocess_x,
                N_test=N,
                C_test=C
            ) 
            _, _, _, self.test_dataloader, _ = get_dataloader(
                self.args, self.tabular_data.is_regression, (N_test, C_test), y_test, self.data_preprocess_y.info, is_train=False
            )
            (self.N_test, self.C_test), self.y_test = get_data_by_split((N_test, C_test), y_test, split="test")

    def fit(self, data, train=True, info=None, config=None):
        if self.args.distribute:
            self.args.num_tasks = get_world_size()
            self.args.local_rank = get_rank()
            self.args.device = torch.device("cuda", self.args.local_rank)

        if isinstance(data, tuple) and info is not None:
            self.tabular_data = TabularData(data[0], data[1], data[2], info, None)
        else:
            self.tabular_data = data

        if config is not None:
            self.reset_stats_withconfig(config)

        self.data_preprocessing(is_train=True)
        self.get_model()
        self.model.to(self.args.train_dtype)
        self.model.to(self.args.device)
        if self.args.distribute:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.model_config["training"]["lr"], 
            weight_decay=self.args.model_config["training"]["weight_decay"]
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
            # print(f"Epoch: {epoch}, Time cost: {cost}")
            print(f"dataset: {self.args.dataset_name}, Epoch: {epoch}, Time cost: {cost}")
            if not self.continue_training:
                break
        save_ddp(self.model, os.path.join(self.args.save_path, f"epoch-last-{self.args.seed}.pth"))
        return time_cost

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

    def train_epoch(self, epoch):
        self.model.train()
        if self.args.distribute:
            self.train_dataloader.sampler.set_epoch(epoch)
        tl = Averager()
        for i, (X, y) in enumerate(self.train_dataloader, 1):
            self.train_step = self.train_step + 1
            
            X_num, X_cat = get_num_and_cat_feats(X, self.N, self.C)

            loss = self.criterion(self.model(X_num, X_cat), y)

            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (i-1) % 50 == 0 or i == len(self.train_dataloader):
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch}, train {i}/{len(self.train_dataloader)}, loss={loss.item():.4f} lr={lr:.4g}")
            del loss
        tl = tl.item()
        self.train_states.train_loss.append(tl)    

    def validate(self, epoch):
        print(f"best epoch {self.train_states.best_epoch}, best val res={self.train_states.best_result:.4f}")
        
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_dataloader)):
                X_num, X_cat = get_num_and_cat_feats(X, self.N, self.C)                   

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
            save_ddp(self.model,os.path.join(self.args.save_path, f"best-val-{self.args.seed}.pth"))
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 30:
                self.continue_training = False
        if self.args.distribute:
            if dist.get_rank() == 0:
                torch.save(self.train_states, os.path.join(self.args.save_path, "train_states"))   
        else:
            torch.save(self.train_states, os.path.join(self.args.save_path, "train_states")) 
