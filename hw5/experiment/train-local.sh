#!/bin/bash
exp_path="/home/sudamonster/course/en600.468/hw5/experiment"
data_path="${exp_path}/data/hw5"
model_path="${exp_path}/model"
pretrained_model="/home/sudamonster/course/mt-hw5-data/model.param"

tool=".."

source activate NeuralHawkes

python ${tool}/train.py \
    --data_file ${data_path} \
    --model_file ${model_path} \
    --optimizer Adam \
    --load_pretrain ${pretrained_model} \
    --learning_rate 0.01
