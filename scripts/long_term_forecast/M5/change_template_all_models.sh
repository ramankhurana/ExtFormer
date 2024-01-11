#!/bin/bash

pred_len_values=(12 24 48 96 144 192 336 720)
seq_len_values=(12 24 48 96 192)
model_names=("Autoformer" "DLinear" "FEDformer")

for model_name in "${model_names[@]}"
do
    for seq_len_value in "${seq_len_values[@]}"
    do
        for pred_len_value in "${pred_len_values[@]}"
        do
            if [ $pred_len_value -le $seq_len_value ]; then
                script_name="run_${model_name}_seq${seq_len_value}_pred${pred_len_value}.sh"

                # Determine label_len based on pred_len_value
                if [ $pred_len_value -lt 48 ]; then
                    label_len_value=$((pred_len_value / 2))
                else
                    label_len_value=48
                fi

                # Use sed to replace SEQ_LEN, PRED_LEN, LABEL_LEN, and MODEL_NAME in the template
                sed -e "s/SEQ_LEN/${seq_len_value}/g" -e "s/PRED_LEN/${pred_len_value}/g" -e "s/LABEL_LEN/${label_len_value}/g" -e "s/MODEL_NAME/${model_name}/g" template_model.sh > $script_name
                chmod +x $script_name
            fi
        done
    done
done
