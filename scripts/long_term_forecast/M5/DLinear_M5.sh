export CUDA_VISIBLE_DEVICES=2

model_name=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/M5/ \
  --data_path sales_1000_Random_columns.csv \
  --model_id M5_96_96 \
  --model $model_name \
  --data M5 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
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
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/M5/ \
  --data_path sales_1000_Random_columns.csv \
  --model_id M5_96_96 \
  --model $model_name \
  --data M5 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --itr 1 \
  --static static1

model_name=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/M5/ \
  --data_path sales_1000_Random_columns.csv \
  --model_id M5_96_96 \
  --model $model_name \
  --data M5 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --itr 1 \
  --static static2

model_name=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/M5/ \
  --data_path sales_1000_Random_columns.csv \
  --model_id M5_96_96 \
  --model $model_name \
  --data M5 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --itr 1 \
  --static static4

model_name=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/M5/ \
  --data_path sales_1000_Random_columns.csv \
  --model_id M5_96_96 \
  --model $model_name \
  --data M5 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --itr 1 \
  --static static6



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/M5/ \
  --data_path sales_1000_Random_columns.csv \
  --model_id M5_96_96 \
  --model $model_name \
  --data M5 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --itr 1 \
  --static static7




#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_192 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 192 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_336 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 336 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_720 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 720 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1
