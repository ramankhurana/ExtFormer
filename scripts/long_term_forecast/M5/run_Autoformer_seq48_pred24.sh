#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

model_name=Autoformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /nfs/home/khurana/dataset/M5/ \
  --data_path sales_1000_Random_columns.csv \
  --model_id M5_48_24 \
  --model $model_name \
  --data M5 \
  --features M \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --itr 1 \
  --static NONE



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /nfs/home/khurana/dataset/M5/ \
  --data_path sales_1000_Random_columns.csv \
  --model_id M5_48_24 \
  --model $model_name \
  --data M5 \
  --features M \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --itr 1 \
  --static static4
