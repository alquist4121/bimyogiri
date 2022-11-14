#!/bin/bash

# MODEL_NAME=rinna/japanese-gpt-1b
MODEL_NAME=rinna/japanese-gpt2-medium
TRAIN_FILE=data/dummy_odai.txt
OUTPUT_DIR=output/$MODEL_NAME

python3 scripts/run_clm.py \
    --model_name_or_path $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --do_train \
    --do_eval \
    --block_size 32 \
    --num_train_epochs 10 \
    --save_total_limit 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --use_fast_tokenizer False \
    --seed 888 \
    --fp16

python3 scripts/generate.py -d /home/u00691/bimyogiri/output/rinna/japanese-gpt2-medium