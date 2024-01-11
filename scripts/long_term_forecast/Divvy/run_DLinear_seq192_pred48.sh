#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

model_name=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /nfs/home/khurana/dataset/DivvyBikes/ \
  --data_path df2021-8-9-10_VAR_nextdayAshrdata.csv \
  --model_id Divvy_192_48 \
  --model $model_name \
  --data Divvy \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 48 \
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
  --root_path /nfs/home/khurana/dataset/DivvyBikes/ \
  --data_path df2021-8-9-10_VAR_nextdayAshrdata.csv \
  --model_id Divvy_192_48 \
  --model $model_name \
  --data Divvy \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --itr 1 \
  --static static4
