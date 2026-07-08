import os

import torch
import torch.distributed as dist


def dist_barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()
    else:
        pass

def dist_destroy():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()
    else:
        pass

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu_id = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distribute = False
        return

    args.distribute = True

    torch.cuda.set_device(args.gpu_id)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def is_main_process():
    return get_rank() == 0


def print_ddp(x):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(x)
    else:
        print(x)

def save_ddp(model,path):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            if os.path.exists(path):
                os.remove(path)
            torch.save(dict(params=model.module.state_dict()), path)
    else:
        torch.save(dict(params=model.state_dict()), path)

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
