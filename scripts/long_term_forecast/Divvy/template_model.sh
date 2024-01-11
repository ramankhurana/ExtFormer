#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

model_name=MODEL_NAME

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /nfs/home/khurana/dataset/DivvyBikes/ \
  --data_path df2021-8-9-10_VAR_nextdayAshrdata.csv \
  --model_id Divvy_SEQ_LEN_PRED_LEN \
  --model $model_name \
  --data Divvy \
  --features M \
  --seq_len SEQ_LEN \
  --label_len LABEL_LEN \
  --pred_len PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 200 \
  --dec_in 200 \
  --c_out 200 \
  --des 'Exp' \
  --itr 1 \
  --static NONE



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /nfs/home/khurana/dataset/DivvyBikes/ \
  --data_path df2021-8-9-10_VAR_nextdayAshrdata.csv \
  --model_id Divvy_SEQ_LEN_PRED_LEN \
  --model $model_name \
  --data Divvy \
  --features M \
  --seq_len SEQ_LEN \
  --label_len LABEL_LEN \
  --pred_len PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 200 \
  --dec_in 200 \
  --c_out 200 \
  --des 'Exp' \
  --itr 1 \
  --static static4
