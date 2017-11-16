#!/bin/bash
exp_path="/Users/sudamonster/courses/en600.468/hw5/experiment"
data_path="${exp_path}/data/hw5"
model_path="${exp_path}/model"
pretrained_model="/Users/sudamonster/Downloads/drive-download-20171115T184743Z-001/model.param"

tool=".."

source activate pytorch_27

python ${tool}/train.py \
    --data_file ${data_path} \
    --model_file ${model_path} \
    --optimizer Adam \
    --load_pretrain ${pretrained_model} \
    --learning_rate 0.001
