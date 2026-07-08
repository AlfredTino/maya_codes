import os
import time

import numpy as np
import torch
from sklearn.feature_selection import (mutual_info_classif,
                                       mutual_info_regression)
                                       
from utils.common_utils import Averager
from utils.data_utils import TabularData, TaskType, get_num_and_cat_feats
from utils.ddp_utils import get_rank, get_world_size, save_ddp

from models._base import BaseTabAnalyzer
from models.modeling.modeling_amformer import AMFormerModel
from models.modeling.modeling_autoint import AutoIntModel
from models.modeling.modeling_excelformer import ExcelFormerModel
from models.modeling.modeling_ftt import FTTModel
from models.modeling.modeling_saint import SAINTModel
from models.modeling.modeling_tabtransformer import TabTransformerModel


class AMFormer(BaseTabAnalyzer):
    def __init__(self, args, task_type):
        super().__init__(args, task_type)

    def get_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.model_config["model"]
        self.model = AMFormerModel(
            num_cont=self.d_in,
            num_cate=len(self.categories) if self.categories is not None else 0,
            categories=self.categories,
            out=self.d_out,
            **model_config
        )

class AutoInt(BaseTabAnalyzer):
    def __init__(self, args, task_type):
        super().__init__(args, task_type)

    def get_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.model_config["model"]
        self.model = AutoIntModel(
            d_numerical=self.d_in,
            categories=self.categories,
            d_out=self.d_out,
            **model_config
        )

class ExcelFormer(BaseTabAnalyzer):
    def __init__(self, args, task_type):
        super().__init__(args, task_type)

    def get_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.model_config["model"]
        self.model = ExcelFormerModel(
            d_numerical=self.d_in,
            d_out=self.d_out,
            **model_config
        )

    def fit(self, data, train=True, info=None, config=None):
        if self.args.distribute:
            self.args.num_tasks = get_world_size()
            self.args.local_rank = get_rank()
            self.args.device = torch.device("cuda", self.args.local_rank)

        if isinstance(data, tuple) and info is not None:
            self.tabular_data = TabularData(data[0], data[1], data[2], info, None)
        else:
            self.tabular_data = data

        mi_func = mutual_info_regression if self.tabular_data.is_regression else mutual_info_classif
        self.data_preprocessing(is_train=True)

        if config is not None:
            self.reset_stats_withconfig(config)

        mi_scores = mi_func(self.N["train"].cpu(), self.y["train"].cpu())
        mi_ranks = np.argsort(-mi_scores)
        self.sorted_mi_scores = torch.from_numpy(mi_scores[mi_ranks] / mi_scores.sum()).to(torch.float64).to(self.args.device)
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
        self.mix_type = self.args.model_config["training"]["mix_type"]

        if not train:
            return

        time_cost = 0
        for epoch in range(self.args.max_epoch):
            start = time.time()
            self.train_epoch(epoch)
            self.validate(epoch)
            cost = time.time() - start
            time_cost += cost
            print(f"Epoch: {epoch}, Time cost: {cost}")
            if not self.continue_training:
                break
        save_ddp(self.model, os.path.join(self.args.save_path, f"epoch-last-{self.args.seed}.pth"))
        return time_cost

    def train_epoch(self, epoch):
        self.model.train()
        if self.args.distribute:
            self.train_dataloader.sampler.set_epoch(epoch)
        tl = Averager()
        for i, (X, y) in enumerate(self.train_dataloader, 1):
            self.train_step = self.train_step + 1
            X_num, X_cat = get_num_and_cat_feats(X, self.N, self.C)

            if self.mix_type == "none":
                loss = self.criterion(self.model(X_num, X_cat,mix_up=False), y)
            else:
                preds, feat_masks, shuffled_ids = self.model(X_num, X_cat,mix_up=True)
                if self.mix_type == "feat_mix":
                    lambdas = (self.sorted_mi_scores * feat_masks).sum(1) # bs
                    lambdas2 = 1 - lambdas
                elif self.mix_type == "hidden_mix":
                    lambdas = feat_masks
                    lambdas2 = 1 - lambdas
                elif self.mix_type == "niave_mix":
                    lambdas = feat_masks
                    lambdas2 = 1 - lambdas
                if self.task_type == TaskType.REGRESSION:
                    mix_y = lambdas * y + lambdas2 * y[shuffled_ids]
                    loss = self.criterion(preds, mix_y)
                else:
                    loss = lambdas * self.criterion(preds, y) + lambdas2 * self.criterion(preds, y[shuffled_ids])
                    loss = loss.mean()
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

class FTT(BaseTabAnalyzer):
    def __init__(self, args, task_type):
        super().__init__(args, task_type)

    def get_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.model_config["model"]
        self.model = FTTModel(
            d_numerical=self.d_in,
            categories=self.categories,
            d_out=self.d_out,
            **model_config
        )

class SAINT(BaseTabAnalyzer):
    def __init__(self, args, task_type):
        super().__init__(args, task_type)
    
    def get_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.model_config["model"]
        self.model = SAINTModel(
            categories=self.categories,
            num_continuous=self.d_in,
            y_dim=self.d_out,
            **model_config
        )

class TabTransformer(BaseTabAnalyzer):
    def __init__(self, args, task_type):
        super().__init__(args, task_type)

    def get_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.model_config["model"]
        self.model = TabTransformerModel(
            categories=self.categories,
            num_continuous=self.d_in,
            dim_out=self.d_out,
            mlp_act=torch.nn.ReLU(),
            mlp_hidden_mults=(4, 2),
            **model_config
        )
