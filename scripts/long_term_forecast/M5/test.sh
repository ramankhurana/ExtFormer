#!/bin/bash

model_name=FEDformer
seq_len_values=(12 24 48 96 192 336)

for seq_len_value in "${seq_len_values[@]}"
do
    script_name="run_${model_name}_seq${seq_len_value}.sh"
    cat > $script_name << EOF
#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /nfs/home/khurana/dataset/M5/ \
  --data_path sales_1000_Random_columns.csv \
  --model_id M5_${seq_len_value}_${seq_len_value} \
  --model $model_name \
  --data M5 \
  --features M \
  --seq_len $seq_len_value \
  --label_len 48 \
  --pred_len $seq_len_value \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1000 \
  --dec_in 1000 \
  --c_out 1000 \
  --des 'Exp' \
  --itr 1 \
  --static NONE
EOF

    chmod +x $script_name
done
