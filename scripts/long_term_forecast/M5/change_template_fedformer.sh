#!/bin/bash

seq_len_values=(12 24 48 96 192 336)

for seq_len_value in "${seq_len_values[@]}"
do
    for pred_len_value in "${seq_len_values[@]}"
    do
        if [ $pred_len_value -le $seq_len_value ]; then
            script_name="run_${model_name}_seq${seq_len_value}_pred${pred_len_value}.sh"
            sed -e "s/SEQ_LEN/${seq_len_value}/g" -e "s/PRED_LEN/${pred_len_value}/g" template_fedformer.sh > $script_name
            chmod +x $script_name
        fi
    done
done

