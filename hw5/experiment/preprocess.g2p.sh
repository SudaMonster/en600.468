#!/bin/bash
origin_data="/export/b18/xma/course/machine_translation/g2p-data"
nmt_path=".."
mkdir -p data

source activate NeuralHawkes

python ${nmt_path}/preprocess.py \
    --train_file ${origin_data}/cmudict.words.trn \
    --dev_file ${origin_data}/cmudict.words.dev \
    --test_file ${origin_data}/cmudict.words.tst \
    --vocab_file ${origin_data}/words.vocab \
    --data_file ./data/g2p.words

python ${nmt_path}/preprocess.py \
    --train_file ${origin_data}/cmudict.phoneme.trn \
    --dev_file ${origin_data}/cmudict.phoneme.dev \
    --test_file ${origin_data}/cmudict.phoneme.tst \
    --vocab_file ${origin_data}/phoneme.vocab \
    --data_file ./data/g2p.phoneme \
    --charniak 
