#!/bin/bash

tuning_methods=("lora" "ia3" "qlora-8bit" "prefix-tuning" "qlora-4bit")
loss_modes=("learn_added_penalize_deleted" "learn_added_unlearn_deleted" "learn_added" "all_after" "unlearn_deleted")
model_types=("multi" "mono" "nl")

# 使用for循环遍历所有调优方法
for method in "${tuning_methods[@]}"
do
  for loss_mode in "${loss_modes[@]}"
  do
    echo "Running training with tuning method: $method and loss mode: $loss_mode"
    CUDA_VISIBLE_DEVICES=1 python main.py \
      --model_name_or_path Salesforce/codegen-2B-multi \
      --dataset sstubs \
      --tuning_method $method \
      --num_epochs 5 \
      --batch_size 4 \
      --gradient_accumulation_steps 2 \
      --learning_rate 3e-4 \
      --lora_r 8 \
      --lora_alpha 16 \
      --do_train \
      --loss_mode $loss_mode
  done
done




for method in "${tuning_methods[@]}"
do
  for loss_mode in "${loss_modes[@]}"
  do
    echo "Running training with tuning method: $method and loss mode: $loss_mode"
    CUDA_VISIBLE_DEVICES=1 python main.py \
      --model_name_or_path Salesforce/codegen-2B-mono \
      --dataset sstubs \
      --tuning_method $method \
      --num_epochs 5 \
      --batch_size 4 \
      --gradient_accumulation_steps 2 \
      --learning_rate 3e-4 \
      --lora_r 8 \
      --lora_alpha 16 \
      --do_train \
      --loss_mode $loss_mode
  done
done 

# for method in "${tuning_methods[@]}"
# do
#   for loss_mode in "${loss_modes[@]}"
#   do
#     echo "Running training with tuning method: $method and loss mode: $loss_mode"
#   CUDA_VISIBLE_DEVICES=0 python sstubs_eval.py \
#       --model_name_or_path Salesforce/codegen-350M-multi  \
#       --adapter_path runs/checkpoints/$loss_mode/sstubs/codegen-350M-multi_$method \
#       --tuning_method $method
#   done
# done