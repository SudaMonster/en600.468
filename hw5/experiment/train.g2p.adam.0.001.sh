#!/bin/bash
exp_path="/export/b18/xma/course/machine_translation/en600.468/hw5/experiment/"
data_path="${exp_path}/data/g2p"
model_path="${exp_path}/model/model.g2p.adam.0.001"
#pretrained_model="/export/b18/xma/course/machine_translation/mt-hw5-data/model.param"

tool="/export/b18/xma/course/machine_translation/en600.468/hw5"

source activate NeuralHawkes

# Availible GPU
device=`/home/xma/tools/bin/free-gpu`
echo "GPU:${device}"

# Necessary environment variables
LD_LIBRARY_PATH=/export/b18/xma/libs/cudnn-6/cuda/lib64:/opt/NVIDIA/cuda-8/lib64/
CPATH=/export/b18/xma/libs/cudnn-6/cuda/include
LIBRARY_PATH=/export/b18/xma/libs/cudnn-6/cuda/lib64

#mkdir -p model
CUDA_VISIBLE_DEVICES=${device} python ${tool}/train.py \
    --data_file ${data_path} \
    --model_file ${model_path} \
    --src_lang words \
    --trg_lang phoneme \
    --optimizer Adam \
    --learning_rate 0.001 \
    --gpuid 0
