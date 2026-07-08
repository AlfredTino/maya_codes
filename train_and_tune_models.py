import dataclasses as dc
import multiprocessing as mp
import time
import typing as ty
from dataclasses import field

import torch
from tqdm import tqdm

from utils.common_utils import (args_parser, display, get_model_by_name,
                                set_gpu, set_seeds)
from utils.data_utils import TabularData
from utils.ddp_utils import dist_barrier, dist_destroy, init_distributed_mode
from utils.tuning_utils import hyperparameter_tuning_by_optuna


@dc.dataclass
class TrainResults:
    losses: ty.List = field(default_factory=list)
    results: ty.List = field(default_factory=list)
    times: ty.List = field(default_factory=list)
    
    def update_results(self, loss=None, result=None, time=None):
        if loss is not None:
            self.losses.append(loss)
        if result is not None:
            self.results.append(result)
        if time is not None:
            self.times.append(time)

def method_train(result_queue, seed, gpu_id, train_val_data, test_data, info, args, violation_seed_queue):
    set_seeds(seed)
    args.gpu = gpu_id
    model = get_model_by_name(args.model_name)(args, info["task_type"])
    try:
        time_cost = model.fit(train_val_data, info=info)    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            violation_seed_queue.put(seed)
            print(f"Model not available at current seed {args.seed}")
            return 
    test_loss, test_res, metric_name, _ = model.predict(test_data)
    result_queue.put((test_loss, test_res, metric_name, time_cost))
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_results = TrainResults()
    args, model_default_configs, model_tuning_space = args_parser()
    init_distributed_mode(args)
    tabular_data = TabularData.from_dir(args.dataset_path, args.dataset_name)
    train_val_data, test_data, data_info = tabular_data._get_split_data() 
    start_time = time.time()
    if args.tune:
        if len(args.tune_datasets) == 0:
            args = hyperparameter_tuning_by_optuna(args, model_tuning_space, train_val_data, data_info)
        else:
            pass
    
    mp.set_start_method("spawn", force=True)
    gpu_list = args.ddp_gpu_ids
    seed_cnt = 0
    processes = []
    result_queue = mp.Queue()
    violation_seed_queue = mp.Queue()
    for seed in tqdm(range(args.seed_num)):
        args.seed = seed    # update seed  
        if args.multiprocessing:
            gpu_id = gpu_list[seed % len(gpu_list)]
            set_gpu(str(gpu_id))
            p = mp.Process(target=method_train, args=(result_queue, seed, gpu_id, train_val_data, test_data, data_info, args, violation_seed_queue))
            processes.append(p)
            p.start()
        else:
            set_seeds(args.seed)
            model = get_model_by_name(args.model_name)(args, data_info["task_type"])
            time_cost = model.fit(tabular_data)
            test_loss, test_res, metric_name, predict_logits = model.predict(test_data)
            train_results.update_results(
                loss=test_loss,
                result=test_res,
                time=time_cost
            )
            dist_barrier()
    if args.multiprocessing:
        for p in processes:
            p.join()
        if not violation_seed_queue.empty():
            if violation_seed_queue.qsize() == args.seed_num:
                raise RuntimeError("This error means no one seed can be ran in single gpu, please use distribute mode")
        loop_cnt = 0
        while not violation_seed_queue.empty():
            for i in range(violation_seed_queue.qsize()):
                seed = violation_seed_queue.get()
                args.seed = seed    # update seed  
                gpu_id = gpu_list[seed % len(gpu_list)]
                set_gpu(str(gpu_id))
                p = mp.Process(target=method_train, args=(result_queue, seed, gpu_id, train_val_data, test_data, data_info, args, violation_seed_queue))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            loop_cnt += 1
            if loop_cnt > 5:
                raise RuntimeError("too many loop_cnt, please use distribute mode or add gpu nums")
        if violation_seed_queue.qsize() != 0:
            raise RuntimeError("violation_seed_queue size is not zero, please use distribute mode")
        while not result_queue.empty():
            test_loss, test_res, metric_name, time_cost = result_queue.get()
            train_results.update_results(
                loss=test_loss,
                result=test_res,
                time=time_cost
            ) 

    display(args, data_info, metric_name, train_results)
    dist_destroy()
    end_time = time.time()
    time_cos = end_time - start_time
    print(f"Total time cost is {time_cos}")
