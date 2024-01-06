#!/bin/bash

seq_len_values=(12 24 48 96 192 336)

for seq_len_value in "${seq_len_values[@]}"
do
    script_name="run_${model_name}_seq${seq_len_value}.sh"
    sed "s/SEQ_LEN/${seq_len_value}/g" template_fedformer.sh > $script_name
    chmod +x $script_name
done
