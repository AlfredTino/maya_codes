#!/bin/bash
datasets=(
    blastchar
    qsar_mod
    seismic+bumps
    shrutime
)
    
models_using_indices=(
    amformer
    autoint
    ftt
    maya
    saint
    tabtransformer
)

models_using_catboost=(
    excelformer
)

work_dir=$(cd "$(dirname "$0")" && pwd)
dataset_path="$work_dir/data/part_i"

GPU_IDS="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
GPU_NUMS=${#GPU_ARRAY[@]}

echo "GPU_NUMS: $GPU_NUMS"
echo "GPU_IDS: $GPU_IDS"

DISTRIBUTE_FLAG=""
GPU_IDS_FLAG=""
MASTER_PORT=""
SINGAL_FLAGE=""
SINGAL_GPU_ID="0"
TUNE_FLAG="--tune"
USING_DISTRIBUTE=0
USING_MULTIPROCESSING=1

if [ $USING_DISTRIBUTE -eq 1 ]; then
    COMMAND_FLAG="torchrun --nproc_per_node ${GPU_NUMS} --nnodes=1 "
    DISTRIBUTE_FLAG="--distribute"
    GPU_IDS_FLAG="--ddp_gpu_ids $GPU_IDS"
    MASTER_PORT="--master_port 29808"
    echo "Using distribute training"
elif [ $USING_MULTIPROCESSING -eq 1 ]; then
    COMMAND_FLAG="python"
    DISTRIBUTE_FLAG="--multiprocessing"
    GPU_IDS_FLAG="--ddp_gpu_ids $GPU_IDS"
    echo "Using multiprocessing for training not using distribute"
else
    COMMAND_FLAG="python"
    echo "Using singal gpu ${SINGAL_GPU_ID} for training"
    SINGAL_FLAGE="--gpu ${SINGAL_GPU_ID}"
fi

for dataset in "${datasets[@]}"; do
    for model in "${models_using_indices[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU_IDS} ${COMMAND_FLAG} ${MASTER_PORT} ./train_and_tune_models.py ${DISTRIBUTE_FLAG} ${SINGAL_FLAGE} ${TUNE_FLAG} ${GPU_IDS_FLAG} --dataset_name $dataset --dataset_path $dataset_path --model_name $model --cat_policy indices > "./logs/${dataset}-${model}${TUNE_FLAG}.txt"
    done
    for model in "${models_using_catboost[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU_IDS} ${COMMAND_FLAG} ${MASTER_PORT} ./train_and_tune_models.py ${DISTRIBUTE_FLAG} ${SINGAL_FLAGE} ${TUNE_FLAG} ${GPU_IDS_FLAG} --dataset_name $dataset --dataset_path $dataset_path --model_name $model --cat_policy catboost > "./logs/${dataset}-${model}${TUNE_FLAG}.txt"
    done
done
