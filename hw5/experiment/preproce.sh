#!/bin/bash
origin_data="../../../mt-hw5-data"
nmt_path=".."
mkdir -p data

python ${nmt_path}/preprocess.py \
    --train_file ${origin_data}/trn.de \
    --dev_file ${origin_data}/dev.de \
    --test_file ${origin_data}/devtest.de \
    --vocab_file ${origin_data}/model.src.vocab \
    --data_file ./data/hw5.de

python ${nmt_path}/preprocess.py \
    --train_file ${origin_data}/trn.en \
    --dev_file ${origin_data}/dev.en \
    --test_file ${origin_data}/devtest.fake.en \
    --vocab_file ${origin_data}/model.trg.vocab \
    --data_file ./data/hw5.en
