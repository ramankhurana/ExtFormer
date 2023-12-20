export CUDA_VISIBLE_DEVICES=2

model_name=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/DivvyBikes/ \
  --data_path df2021-8-9-10_VAR_nextdayAshrdata.csv \
  --model_id Divvy_96_96 \
  --model $model_name \
  --data Divvy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 200 \
  --dec_in 200 \
  --c_out 200 \
  --des 'Exp' \
  --itr 1 \
  --static static1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/DivvyBikes/ \
  --data_path df2021-8-9-10_VAR_nextdayAshrdata.csv \
  --model_id Divvy_96_96 \
  --model $model_name \
  --data Divvy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 200 \
  --dec_in 200 \
  --c_out 200 \
  --des 'Exp' \
  --itr 1 \
  --static static2


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/DivvyBikes/ \
  --data_path df2021-8-9-10_VAR_nextdayAshrdata.csv \
  --model_id Divvy_96_96 \
  --model $model_name \
  --data Divvy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 200 \
  --dec_in 200 \
  --c_out 200 \
  --des 'Exp' \
  --itr 1 \
  --static static4


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/DivvyBikes/ \
  --data_path df2021-8-9-10_VAR_nextdayAshrdata.csv \
  --model_id Divvy_96_96 \
  --model $model_name \
  --data Divvy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 200 \
  --dec_in 200 \
  --c_out 200 \
  --des 'Exp' \
  --itr 1 \
  --static static6


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../JupyterNotebooks/Comparison_Multivariate/dataset/DivvyBikes/ \
  --data_path df2021-8-9-10_VAR_nextdayAshrdata.csv \
  --model_id Divvy_96_96 \
  --model $model_name \
  --data Divvy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 200 \
  --dec_in 200 \
  --c_out 200 \
  --des 'Exp' \
  --itr 1 \
  --static static7

