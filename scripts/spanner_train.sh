#!/bin/bash


cd /mnt/nas-alinlp/qinghui.zz/projects/linkner/A_Linkner
DATA_DIR="Your training data path"
BERT_CONFIG_DIR="Your pre-trained model path"
MODEL_SAVE_DIR="The path where you save the trained model"

python main.py \
    --loss 'ce' \
    --test_mode 'ori' \
    --etrans_func "softmax" \
    --data_dir "{DATA_DIR}" \
    --state "train" \
    --bert_config_dir "{BERT_CONFIG_DIR}" \
    --batch_size 64 \
    --max_spanLen 5 \
    --n_class 5 \
    --model_save_dir "{MODEL_SAVE_DIR}" \
    --iteration 100
