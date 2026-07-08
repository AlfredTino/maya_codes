import bisect
import json
import logging
import multiprocessing as mp
import os
import threading
from datetime import datetime

import numpy as np
import optuna
import optuna.samplers
import optuna.trial

from utils.common_utils import get_model_by_name, set_gpu, set_seeds


def adjust_args_batch_size(args):
    if args.distribute:
        ddp_gpu_num = len(args.ddp_gpu_ids)
        if args.batch_size % ddp_gpu_num == 0:
            args.batch_size = int(args.batch_size / ddp_gpu_num)
        else:
            original_batch = args.batch_size
            args.batch_size = args.batch_size // ddp_gpu_num
            real_batch_size = args.batch_size * ddp_gpu_num
            print(f"giving batch size {original_batch} can't be divised by gpu num {ddp_gpu_num}, batch size is set to {real_batch_size}")

    return args

def early_stop_seed_train(current_trial_res_list, best_res, total_seed_num, direction):
    if total_seed_num < 3:
        return False
    current_seed_num = len(current_trial_res_list)
    if current_seed_num < (total_seed_num - 2):
        return False
    if current_seed_num == total_seed_num:
        return False
    current_trial_res_mean = np.mean(current_trial_res_list)
    current_trial_res_var = np.std(current_trial_res_list)
    if current_seed_num == (total_seed_num - 2):
        scope = 3
    elif current_seed_num == (total_seed_num - 1):
        scope = 1.5
    current_trial_res_scope = current_trial_res_var * scope
    if direction == "minimize":
        if (current_trial_res_mean - current_trial_res_scope) > best_res:
            print("*********early stop.*********")
            return True
    elif direction == "maximize":
        if (current_trial_res_mean + current_trial_res_scope) < best_res:
            print("*********early stop.*********")
            return True

    return False

def find_insert_position_desc(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] > target:
            left = mid + 1
        elif arr[mid] < target:
            right = mid - 1
        else:
            return mid

    return left

def hyperparameter_tuning_by_optuna(args, tuning_space, train_val_data, info):
    shared_list = list(range(len(args.ddp_gpu_ids))) 
    list_lock = threading.Lock()
    current_trials_best_res=0
    def objective(trial):
        config = {}
        merge_sampled_parameters(
            config, sample_parameters(trial, tuning_space, config)
        )    

        config = set_default_parameters_by_model(args.model_name, config)
        
        trial_configs.append(config)
        try:
            if type(train_val_data) is dict:
                pass
            else:
                if args.multiprocessing:
                    mp.set_start_method("spawn", force=True)
                    gpu_list = args.ddp_gpu_ids
                    processes = []
                    result_queue = mp.Queue()
                    best_res_list = []
                    nonlocal current_trials_best_res 
                    while True:
                        with list_lock: 
                            if shared_list: 
                                num = shared_list.pop(0) 
                                print(f"trial {trial.number} took {num}") 
                                break
                            else: 
                                continue
                    gpu_id = gpu_list[num % len(gpu_list)]
                    p = mp.Process(target=train_mul_seed,
                                   args=(result_queue, gpu_id, train_val_data, info, config, args, current_trials_best_res, direction))
                    p.start()
                    p.join()
                    while not result_queue.empty():
                        best_res_list.append(result_queue.get())
                    if len(best_res_list) != 1:
                        print(f"[ERROR !!!!!] len(best_res_list) != 1; got best_res_list is {best_res_list}")
                        best_res = 0 if direction == "maximize" else float("100000")
                    else:
                        best_res = best_res_list[0]
                    with list_lock: 
                        if (direction=="minimize" and best_res < current_trials_best_res) or (direction=="maximize" and best_res > current_trials_best_res):
                            current_trials_best_res = best_res
                        shared_list.append(num) 
                        print(f"trial {trial.number} put back {num}")

                    return best_res
                else:
                    if "batch_size" in config["model"]:
                        model.args.batch_size = config["model"]["batch_size"]
                        model.args = adjust_args_batch_size(model.args)
                    model.fit(train_val_data, train=True, info=info, config=config)    
                    return model.train_states.best_result
        except Exception as e:
            print(e)
            return 1e9 if info["task_type"] == "regression" else 0.0
    now = datetime.now()
    current_time = f"{now.month}.{now.day}.{now.hour}.{now.minute}"
    log_file = os.path.join(args.save_path, f"{args.model_name}_tuned_logs_{current_time}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler() 
    if os.path.exists(os.path.join(args.save_path, f"{args.model_name}-tuned.json")):
        with open(os.path.join(args.save_path, f"{args.model_name}-tuned.json"), "rb") as f:
            args.model_config = json.load(f)
    else:
        # get data property
        if len(args.tune_datasets) > 0:
            pass
        else:
            if info["task_type"] == "regression":
                direction = "minimize"
                for key in tuning_space["model"].keys():
                    if "dropout" in key and "?" not in tuning_space["model"][key][0]:
                        tuning_space["model"][key][0] = "?" + tuning_space["model"][key][0]
                        tuning_space["model"][key].insert(1, 0.0)
            else:
                direction = "maximize"  
            if not args.multiprocessing:
                model = get_model_by_name(args.model_name)(args, info["task_type"])      

        trial_configs = []
        current_trials_best_res=0 if direction == "maximize" else float("100000")
        study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=0),
            )
        n_jobs=1
        if args.multiprocessing:
            n_jobs=len(args.ddp_gpu_ids)
        study.optimize(
            objective,
            **{"n_trials": args.n_trials},
            show_progress_bar=True,
            n_jobs=n_jobs
        ) 
        # get best configs
        best_trial_id = study.best_trial.number
        # update config files        
        print("Best Hyper-Parameters")
        print(trial_configs[best_trial_id])
        args.model_config = trial_configs[best_trial_id]
        if "batch_size" in args.model_config["model"]:
            args.batch_size = args.model_config["model"]["batch_size"]
            args = adjust_args_batch_size(args)
        with open(os.path.join(args.save_path, f"{args.model_name}-tuned-{current_time}.json"), "w") as f:
            json.dump(args.model_config, f, sort_keys=True, indent=4)
    return args

def merge_sampled_parameters(config, sampled_parameters):
    for k, v in sampled_parameters.items():
        if isinstance(v, dict):
            merge_sampled_parameters(config.setdefault(k, {}), v)
        else:
            config[k] = v

def sample_parameters(trial, space, base_config):
    def get_distribution(distribution_name):
        return getattr(trial, f"suggest_{distribution_name}")

    result = {}
    for label, subspace in space.items():
        if isinstance(subspace, dict):
            result[label] = sample_parameters(trial, subspace, base_config)
        else:
            assert isinstance(subspace, list)
            distribution, *args = subspace

            if distribution.startswith("?"):
                default_value = args[0]
                result[label] = (
                    get_distribution(distribution.lstrip("?"))(label, *args[1:])
                    if trial.suggest_categorical(f"optional_{label}", [False, True])
                    else default_value
                )

            elif distribution == "float":
                result[label] = trial.suggest_float(label, low=args[0], high=args[1], step=args[2], log=args[3])

            else:
                result[label] = get_distribution(distribution)(label, *args)

    return result

def set_default_parameters_by_model(model_name, config):
    if model_name == "amformer":
        config["model"]["heads"] = 8
        config["model"]["groups"] = [54, 54, 54, 54]
        config["model"]["sum_num_per_group"] = [32, 16, 8, 4]
        config["model"]["prod_num_per_group"] = [4, 4, 4, 4]
        config["model"]["cluster"] =  True
        config["model"]["target_mode"] = "mix"
        config["model"]["token_descent"] = False

    elif model_name == "autoint":
        config["model"]["prenormalization"] = False
        config["model"]["initialization"] = "xavier"
        config["model"]["activation"] = "relu"
        config["model"]["n_heads"] = 8
        config["model"]["d_token"] = 64
        config["model"]["kv_compression"] = None
        config["model"]["kv_compression_sharing"] = None

    elif model_name == "excelformer":
        config["model"]["prenormalization"] = False
        config["model"]["kv_compression"] = None
        config["model"]["kv_compression_sharing"] = None
        config["model"]["token_bias"] = True
        config["model"]["init_scale"] = 0.01
        config["model"]["n_heads"] = 8

    elif model_name == "ftt":
        config["model"]["prenormalization"] = False
        config["model"]["initialization"] = "xavier"
        config["model"]["activation"] = "reglu"
        config["model"]["n_heads"] = 8
        config["model"]["d_token"] = 64
        config["model"]["token_bias"] = True
        config["model"]["kv_compression"] = None
        config["model"]["kv_compression_sharing"] = None

    elif model_name == "maya":
        config["model"]["label_nums"] = 1
        config["model"]["skip_first_norm"] = False
        config["model"]["last_mlp_skip"] = True
        config["model"]["using_attn_norm"] = True
        config["model"]["using_encoder_decoder_arch"] = True
        config["model"]["decoder_configs"]["decoder_layer_configs"]["using_scaling_L2"] = False
        config["model"]["decoder_configs"]["decoder_layer_configs"]["decoder_hidden_size"] = config["model"]["hidden_size"]

    return config

def train_mul(rank_list, data_rank, gpu_rank, model, train_val_data, info, sorted_result, config):
    set_gpu(str(gpu_rank))
    model.fit(train_val_data, train=True, info=info, config=config)
    if info["task_type"] == "regression":
        result = bisect.bisect_right(sorted_result, model.train_states.best_result)
    else:
        result = find_insert_position_desc(sorted_result, model.train_states.best_result)
    print(f"dataset {data_rank} rank list is {rank_list} current rank is {result}")
    rank_list.put((data_rank, result))

def train_mul_seed(res_queue, gpu_id, train_val_data, info, config, args, current_trials_best_res, direction):
    set_gpu(str(gpu_id))
    best_res_list = []
    for seed in range(args.tuning_seed_nums):
        args.seed = seed
        set_seeds(seed)
        model = get_model_by_name(args.model_name)(args, info["task_type"])
        model.args.gpu = gpu_id
        if "batch_size" in config["model"]:
            model.args.batch_size = config["model"]["batch_size"]
        try:
            model.fit(train_val_data, train=True, info=info, config=config)
        except Exception as e:
            print(e)
            if "CUDA out of memory" in e:
                print("gpu memory is not enough for this configuration !!!")
            return
        print(f"===========finish seed:{seed}, gpu id: {gpu_id}, result is {model.train_states.best_result}=============")
        best_res_list.append(model.train_states.best_result)
        if early_stop_seed_train(current_trial_res_list=best_res_list,
                                 best_res=current_trials_best_res, 
                                 total_seed_num=args.tuning_seed_nums,
                                 direction=direction):
            break
    best_res = np.mean(best_res_list)
    res_queue.put(best_res)
    print(f"=================best list:{best_res_list}, gpu id: {gpu_id}==================")
