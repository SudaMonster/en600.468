#!/bin/bash
origin_data="/Users/sudamonster/Downloads/drive-download-20171115T184743Z-001"
nmt_path=".."
mkdir -p data

source activate pytorch_27

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
    --data_file ./data/hw5.en \
    --charniak 
