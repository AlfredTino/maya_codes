import dataclasses as dc
import json
import os
import typing as ty
from copy import deepcopy
from pathlib import Path

import category_encoders
import numpy as np
import sklearn.preprocessing
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ArrayDict = ty.Dict[str, np.ndarray]

@dc.dataclass
class DataPreprocessX:
    num_new_value: ty.Optional[np.ndarray]
    cat_new_value: ty.Optional[str]
    normalizer: ty.Any
    ord_encoder: ty.Any
    mode_values: np.ndarray
    cat_encoder: ty.Optional[ty.Any]

@dc.dataclass
class DataPreprocessY:
    info: ty.Dict[str, ty.Any]
    encoder: ty.Optional[ty.Any]

@dc.dataclass
class TabularData:
    N: ty.Optional[ArrayDict]
    C: ty.Optional[ArrayDict]
    y: ArrayDict
    info: ty.Dict[str, ty.Any]
    folder: ty.Optional[Path]

    @classmethod
    def from_dir(cls, dataset_path, dataset_name) -> "TabularData":
        dir_ = Path(os.path.join(os.path.dirname(__file__), "..", dataset_path, dataset_name))

        def load(item) -> ArrayDict:
            return {
                x: ty.cast(np.ndarray, np.load(dir_ / f"{item}_{x}.npy", allow_pickle=True))
                for x in ["train", "val", "test"]
            }

        return TabularData(
            load("N") if dir_.joinpath("N_train.npy").exists() else None,
            load("C") if dir_.joinpath("C_train.npy").exists() else None,
            load("y"),
            load_json(dir_ / "info.json"),
            dir_,
        )

    @property
    def is_binclass(self) -> bool:
        return self.info["task_type"] == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.info["task_type"] == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.info["task_type"] == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return self.info["n_num_features"]

    @property
    def n_cat_features(self) -> int:
        return self.info["n_cat_features"]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def _get_split_data(self):
        if self.info is None:
            raise NameError(f"{self.info} is empty! Please call from_dir first to load a dataset!")
        N_trainval = None if self.N is None else {key: self.N[key] for key in ["train", "val"]} if "train" in self.N and "val" in self.N else None
        N_test = None if self.N is None else {key: self.N[key] for key in ["test"]} if "test" in self.N else None

        C_trainval = None if self.C is None else {key: self.C[key] for key in ["train", "val"]} if "train" in self.C and "val" in self.C else None
        C_test = None if self.C is None else {key: self.C[key] for key in ["test"]} if "test" in self.C else None

        y_trainval = {key: self.y[key] for key in ["train", "val"]}
        y_test = {key: self.y[key] for key in ["test"]} 
        
        train_val_data = (N_trainval, C_trainval, y_trainval)
        test_data = (N_test, C_test, y_test)
        return train_val_data, test_data, self.info

    def size(self, split: str) -> int:
        X = self.N if self.N is not None else self.C
        assert X is not None
        return len(X[split])

    def build_X(
        self,
        normalization: ty.Optional[str],
        cat_policy: str,
        seed: int,
        data_preprocess_x: DataPreprocessX = None,
        y_train=None,
        N_test=None,
        C_test=None
    ) -> ty.Union[ArrayDict, ty.Tuple[ArrayDict, ArrayDict]]:
        if data_preprocess_x is not None:
            # perform preprocessing for test data
            assert(N_test is not None or C_test is not None)
            is_test = True
        else:
            is_test = False

        if self.N:
            # deal with nan in num_features
            if is_test:
                N = deepcopy(N_test)
            else:
                N = deepcopy(self.N)

            if "train" in N.keys():
                if N["train"].ndim == 1:
                    N = {k: v.reshape(-1, 1) for k, v in N.items()}
            else:
                if N["test"].ndim == 1:
                    N = {k: v.reshape(-1, 1) for k, v in N.items()}
            N = {k: v.astype(float) for k,v in N.items()}

            num_nan_masks = {k: np.isnan(v) for k, v in N.items()}
            if any(x.any() for x in num_nan_masks.values()):  # type: ignore[code]
                if data_preprocess_x is None:
                    num_new_value = np.nanmean(N["train"], axis=0)
                else:
                    num_new_value = data_preprocess_x.num_new_value

                for k, v in N.items():
                    num_nan_indices = np.where(num_nan_masks[k])
                    v[num_nan_indices] = np.take(num_new_value, num_nan_indices[1])
            else:
                num_new_value = None
            
            # normalize the num_features
            if data_preprocess_x is None:
                N_train = N["train"].copy()

                if normalization == "standard":
                    normalizer = sklearn.preprocessing.StandardScaler()
                elif normalization == "quantile":
                    normalizer = sklearn.preprocessing.QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=max(min(N["train"].shape[0] // 30, 1000), 10),
                        random_state=seed
                    )
                else:
                    raise_unknown("normalization", normalization)

                normalizer.fit(N_train)
            else:
                normalizer = data_preprocess_x.normalizer
            
            N = {k: normalizer.transform(v) for k, v in N.items()}

        else:
            N = None

        if self.C:
            # deal with nan in cat_features
            if is_test:
                C = deepcopy(C_test)
            else:
                C = deepcopy(self.C)

            if "train" in C.keys():
                if C["train"].ndim == 1:
                    C = {k: v.reshape(-1, 1) for k, v in C.items()}
            else:
                if C["test"].ndim == 1:
                    C = {k: v.reshape(-1, 1) for k, v in C.items()}
            C = {k: v.astype(str) for k,v in C.items()}

            cat_nan_masks = {k: np.isnan(v) if np.issubdtype(v.dtype, np.number) else np.isin(v, ["nan", "NaN", "", None]) for k, v in C.items()}
            if any(x.any() for x in cat_nan_masks.values()):  
                if data_preprocess_x is None:
                    cat_new_value = "___null___"
                else:
                    cat_new_value = data_preprocess_x.cat_new_value
                
                for k, v in C.items():
                    cat_nan_indices = np.where(cat_nan_masks[k])
                    v[cat_nan_indices] = cat_new_value
            else:
                cat_new_value = None

        else:
            C = None

        # encode cat_features
        unknown_value = np.iinfo("int64").max - 3
        if data_preprocess_x is None:
            ord_encoder = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown="use_encoded_value",  # type: ignore[code]
                unknown_value=unknown_value,  # type: ignore[code]
                dtype="int64",  # type: ignore[code]
            ).fit(C["train"])
        else:
            ord_encoder = data_preprocess_x.ord_encoder
        C = {k: ord_encoder.transform(v) for k, v in C.items()}

        if data_preprocess_x is not None:
            mode_values = data_preprocess_x.mode_values
            for column_idx in range(C["test"].shape[1]):
                C["test"][:, column_idx][C["test"][:, column_idx] == unknown_value] = mode_values[column_idx]
        elif "val" in C.keys():
            mode_values = [np.argmax(np.bincount(column[column != unknown_value]))
                        if np.any(column == unknown_value) else column[0]
                        for column in C["train"].T]
            for column_idx in range(C["val"].shape[1]):
                C["val"][:, column_idx][C["val"][:, column_idx] == unknown_value] = mode_values[column_idx]

        if cat_policy == "indices":
            result = (N, C)
            cat_encoder = None
        elif cat_policy == "catboost":
            if data_preprocess_x is None:
                cat_encoder = category_encoders.CatBoostEncoder()
                cat_encoder.fit(C["train"].astype(str), y_train)
            else:
                cat_encoder = data_preprocess_x.cat_encoder
            C = {k: cat_encoder.transform(v.astype(str)).values for k, v in C.items()}
        
        if N is None:
            result = (C, None)
        else:
            result = ({x: np.hstack((N[x], C[x])) for x in N}, None)

        if data_preprocess_x is None:
            data_preprocess_x = DataPreprocessX(
                num_new_value, cat_new_value, normalizer, ord_encoder, mode_values, cat_encoder
            )

        return result, data_preprocess_x

    def build_y(
        self, data_preprocess_y: DataPreprocessY = None, y_test=None
    ) -> ty.Tuple[ArrayDict, ty.Optional[ty.Dict[str, ty.Any]]]:
        if data_preprocess_y is not None:
            # perform preprocessing for test data
            assert(y_test is not None)
            is_test = True
        else:
            is_test = False
        
        if is_test:
            y = deepcopy(y_test)
        else:
            y = deepcopy(self.y)

        if not self.is_regression:
            if data_preprocess_y is None:
                encoder = sklearn.preprocessing.LabelEncoder().fit(y["train"])
            else:
                encoder = data_preprocess_y.encoder
            y = {k:encoder.transform(v) for k, v in y.items()}
            info = {"policy": "none"}
        else:
            y = {k: v.astype(float) for k,v in y.items()}
            if data_preprocess_y is None:
                mean, std = y["train"].mean(), y["train"].std()
            else:
                info = data_preprocess_y.info
                mean, std = info["mean"], info["std"]
            y = {k: (v - mean) / std for k, v in y.items()}
            info = {"policy": "mean_std", "mean": mean, "std": std}
            encoder = None
        
        if data_preprocess_y is None:
            data_preprocess_y = DataPreprocessY(info, encoder)
        return y, data_preprocess_y

class TabularDataset(Dataset):
    def __init__(self, X, Y, y_info, split, dtype):
        X_num, X_cat = X
        self.X_num = X_num[split].to(dtype) if X_num is not None else None
        self.X_cat = X_cat[split] if X_cat is not None else None
        if self.X_cat is not None:
            self.X_cat = self.X_cat if is_integer_tensor(self.X_cat) else self.X_cat.to(dtype)
        if Y[split].dtype == torch.float64:
            Y[split] = Y[split].to(dtype)
        self.Y, self.y_info = Y[split], y_info
        
    def get_dim_in(self):
        return 0 if self.X_num is None else self.X_num.shape[1]

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, i):
        if self.X_num is not None and self.X_cat is not None:
            data = (self.X_num[i], self.X_cat[i])
        elif self.X_cat is not None and self.X_num is None:
            data, label = self.X_cat[i], self.Y[i]
        else:
            data, label = self.X_num[i], self.Y[i]
        label = self.Y[i]
        return data, label

@dc.dataclass
class TaskType:
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

def get_categories(
    X_cat: ty.Optional[ty.Dict[str, torch.Tensor]]
) -> ty.Optional[ty.List[int]]:
    return (
        None
        if X_cat is None
        else [
            len(set(X_cat["train"][:, i].tolist()))
            for i in range(X_cat["train"].shape[1])
        ]
    )

def get_data_by_split(X, y=None, split=None):
    N, C = X
    if N is not None and C is not None:
        if split is not None:
            N_split, C_split = N[split], C[split]
        else:
            N_split, C_split = N, C
    elif N is None and C is not None:
        if split is not None:
            N_split, C_split = None, C[split]
        else:
            N_split, C_split = None, C
    else:
        if split is not None:
            N_split, C_split = N[split], None
        else:
            N_split, C_split = N, None

    if y is not None:
        if split is not None:
            y_split = y[split]
        else:
            y_split = y
    else:
        y_split = None

    X_split = (N_split, C_split)

    return X_split, y_split

def get_dataloader(args, is_regression, X, Y, y_info, is_train):
    device = args.device 
    batch_size = args.batch_size
    X = tuple(None if x is None else to_tensors(x) for x in X)
    Y = to_tensors(Y)

    X = tuple(None if x is None else {k: v.to(device) for k, v in x.items()} for x in X)
    Y = {k: v.to(device) for k, v in Y.items()}

    if X[0] is not None:
        X = ({k: v.double() for k, v in X[0].items()}, X[1])

    if is_regression:
        Y = {k: v.double() for k, v in Y.items()}
    else:
        Y = {k: v.long() for k, v in Y.items()}
    
    loss_fn = (
        F.mse_loss
        if is_regression
        else F.cross_entropy
    )

    if is_train:
        trainset = TabularDataset( X, Y, y_info, "train", args.train_dtype)
        valset = TabularDataset(X, Y, y_info, "val", args.train_dtype)
        if args.distribute:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=args.num_tasks, rank=args.local_rank, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=args.num_tasks, rank=args.local_rank, shuffle=False)
            train_loader = DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset=valset, batch_size=batch_size, sampler=val_sampler)
        else:
            train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)        
            val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, num_workers=0) 
        return X[0], X[1], Y, train_loader, val_loader, loss_fn
    else:
        testset = TabularDataset(X, Y, y_info, "test", args.train_dtype)
        if args.distribute:
            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=args.num_tasks, rank=args.local_rank, shuffle=False)
            test_loader = DataLoader(dataset=testset, batch_size=batch_size, sampler=test_sampler)
        else:
            test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)        
        return X[0], X[1], Y, test_loader, loss_fn

def get_num_and_cat_feats(X, N, C):
    if N is not None and C is not None:
        X_num, X_cat = X[0], X[1]
    elif C is not None and N is None:
        X_num, X_cat = None, X
    else:
        X_num, X_cat = X, None

    return X_num, X_cat

def is_integer_tensor(tensor):  
    if not torch.is_tensor(tensor):  
        return False  

    integer_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]  
    return tensor.dtype in integer_dtypes

def load_json(path):
    return json.loads(Path(path).read_text())

def raise_unknown(unknown_what: str, unknown_value: ty.Any):
    raise ValueError(f"Unknown {unknown_what}: {unknown_value}")

def to_tensors(data: ArrayDict) -> ty.Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v) for k, v in data.items()}
