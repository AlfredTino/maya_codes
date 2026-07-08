import argparse
import json
import os
import random
import warnings

import numpy as np
import torch


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def args_parser():  
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="shrutime", required=False)
    parser.add_argument("--dataset_path", type=str, default="data/part_i", required=False)  
    parser.add_argument("--model_save_path", type=str, default="resulted_models", required=False)

    parser.add_argument("--batch_size", type=int, default=1024)  
    parser.add_argument("--cat_policy", type=str, default="indices", choices=["catboost", "indices"])  
    parser.add_argument("--ddp_gpu_ids", default="0", type=str)
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--distribute", action="store_true", default=False, help="Enabling distributed training")
    parser.add_argument("--gpu_idx", type=str, default="0")
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--model_name", type=str, default="maya",
                        choices=["amformer", "autoint", "excelformer", "ftt", "maya", "saint", "tabtransformer"])
    parser.add_argument("--multiprocessing", action="store_true", default=False)
    parser.add_argument("--n_trials", type=int, default=100)  
    parser.add_argument("--normalization", type=str, default="quantile", choices=["none", "quantile", "standard"])
    parser.add_argument("--seed_num", type=int, default=15)
    parser.add_argument("--train_dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--tune", action="store_true", default=False)
    parser.add_argument("--tune_datasets", type=str, default=None)
    parser.add_argument("--tune_datasets_result", type=str, default="{}")
    parser.add_argument("--tuning_seed_nums", default=1, type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    
    args = parser.parse_args()
    args.ddp_gpu_ids = list(map(int, args.ddp_gpu_ids.split(",")))
    if args.tune_datasets is not None:
        args.tune_datasets = args.tune_datasets.split(",")
    else:
        args.tune_datasets = []
    args.tune_datasets_result = eval(args.tune_datasets_result)
    args.train_dtype = getattr(torch, args.train_dtype) if type(args.train_dtype) is str else args.train_dtype
    ddp_gpu_num = len(args.ddp_gpu_ids)
    if args.distribute:
        if args.batch_size % ddp_gpu_num == 0:
            args.batch_size = int(args.batch_size / ddp_gpu_num)
        else:
            original_batch = args.batch_size
            args.batch_size = args.batch_size // ddp_gpu_num
            real_batch_size = args.batch_size * ddp_gpu_num
            message = f"giving batch size {original_batch} can't be divised by gpu num {ddp_gpu_num}, batch size is set to {real_batch_size}!"
            warnings.warn(message, category=UserWarning)   
    elif args.multiprocessing:
        pass
    else:
        set_gpu(args.gpu_idx)
            
    save_path = "_".join([args.dataset_name, args.model_name, f"epoch{args.max_epoch}", f"bs{args.batch_size}"])
    if args.tune:
        save_path = "Tune_" + save_path
    args.save_path = os.path.join(os.path.dirname(__file__), "..", args.model_save_path, save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)   
    
    model_configs_path = os.path.join(os.path.dirname(__file__), "..", "models/model_configs")
    with open(os.path.join(model_configs_path, f"{args.model_name}_default.json"), "r") as f:
        model_default_configs = json.load(f) 
    if args.tune:
        with open(os.path.join(model_configs_path, f"{args.model_name}_tuning_space.json"), "r") as f:
            model_tuning_space = json.load(f)
    else:
        model_tuning_space = None

    args.model_config = model_default_configs
    
    args.seed = 0
    set_seeds(args.seed)
    if torch.cuda.is_available():     
        torch.backends.cudnn.benchmark = True

    return args, model_default_configs, model_tuning_space   

def display(args, info, metric_name, train_results):
    if not isinstance(metric_name, list):
        metric_name = [metric_name]
    metric_arrays = {name: [] for name in metric_name}  

    for result in train_results.results:
        if not isinstance(result, list):
            result = [result]
        for idx, name in enumerate(metric_name):
            metric_arrays[name].append(result[idx])

    metric_arrays["Time"] = train_results.times
    metric_name = metric_name + ["Time"]

    mean_metrics = {name: np.mean(metric_arrays[name]) for name in metric_name}
    std_metrics = {name: np.std(metric_arrays[name]) for name in metric_name}
    mean_loss = np.mean(np.array(train_results.losses))

    print("-" * 20, "Results", "-" * 20)
    print(f"model: {args.model_name}, run for {args.seed_num} seeds")
    for name in metric_name:
        if info["task_type"] == "regression" and name != "Time":
            formatted_results = ", ".join([f"{e:.8e}" for e in metric_arrays[name]])
            print(f"[{name} All results]: {formatted_results}")
            print(f"[{name} Mean results] = {mean_metrics[name]:.8e} ± {std_metrics[name]:.8e}")
        else:
            formatted_results = ", ".join([f"{e:.8f}" for e in metric_arrays[name]])
            print(f"[{name} All results]: {formatted_results}")
            print(f"[{name} Mean results] = {mean_metrics[name]:.8f} ± {std_metrics[name]:.8f}")

    print(f"[Mean Loss] = {mean_loss:.8e}")

def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model_by_name(model_name):
    from models.compared_methods import (FTT, SAINT, AMFormer, AutoInt, ExcelFormer,
                                  TabTransformer)
    from models.maya import Maya
    NAME_TO_MODEL = {
        "maya": Maya,
        "amformer": AMFormer,
        "autoint": AutoInt,
        "excelformer": ExcelFormer,
        "ftt": FTT,
        "saint": SAINT,
        "tabtransformer": TabTransformer
    }

    return NAME_TO_MODEL[model_name]

def set_gpu(id):
    os.environ["CUDA_VISIBLE_DEVICES"] = id
    print(f"USING GPU{id}")

def set_seeds(base_seed: int, one_cuda_seed: bool = False) -> None:
    assert 0 <= base_seed < 2 ** 32 - 10000
    random.seed(base_seed)
    np.random.seed(base_seed + 1)
    torch.manual_seed(base_seed + 2)
    cuda_seed = base_seed + 3
    if one_cuda_seed:
        torch.cuda.manual_seed_all(cuda_seed)
    elif torch.cuda.is_available():
        # the following check should never succeed since torch.manual_seed also calls
        # torch.cuda.manual_seed_all() inside; but let's keep it just in case
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        # Source: https://github.com/pytorch/pytorch/blob/2f68878a055d7f1064dded1afac05bb2cb11548f/torch/cuda/random.py#L109
        for i in range(torch.cuda.device_count()):
            default_generator = torch.cuda.default_generators[i]
            default_generator.manual_seed(cuda_seed + i)
