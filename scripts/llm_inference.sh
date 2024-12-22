#!/bin/bash

cd /mnt/nas-alinlp/qinghui.zz/projects/linkner/A_Linkner
INPUT_FILE="Your input file path"
SAVE_FILE="Your output save file path"
LLM_CKPT="Your LLM checkpoint path"

python main.py \
    --state llm_classify \
    --input_file "$INPUT_FILE" \
    --save_file "$SAVE_FILE" \
    --linkDataName conll03 \
    --llm_name 'llama' \
    --llm_ckpt "$LLM_CKPT" \
    --selectShot_dir None \
    --shot 0 \
    --threshold 0.4
