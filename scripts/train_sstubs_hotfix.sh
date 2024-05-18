CUDA_VISIBLE_DEVICES=1 python main.py \
  --model_name_or_path Salesforce/codegen-350M-multi \
  --dataset sstubs \
  --tuning_method ia3 \
  --num_epochs 5 \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --do_train

CUDA_VISIBLE_DEVICES=1 python main.py \
  --model_name_or_path Salesforce/codegen-350M-mono \
  --dataset sstubs \
  --tuning_method ia3 \
  --num_epochs 5 \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --do_train

CUDA_VISIBLE_DEVICES=1 python main.py \
  --model_name_or_path Salesforce/codegen-350M-nl \
  --dataset sstubs \
  --tuning_method ia3 \
  --num_epochs 5 \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --do_train


CUDA_VISIBLE_DEVICES=1 python main.py \
  --model_name_or_path Salesforce/codegen-2B-multi \
  --dataset sstubs \
  --tuning_method ia3 \
  --num_epochs 5 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 6e-5 \
  --lora_r 8 \
  --lora_alpha 16 \
  --do_train

CUDA_VISIBLE_DEVICES=1 python main.py \
  --model_name_or_path Salesforce/codegen-2B-mono \
  --dataset sstubs \
  --tuning_method ia3 \
  --num_epochs 5 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 6e-5 \
  --lora_r 8 \
  --lora_alpha 16 \
  --do_train

CUDA_VISIBLE_DEVICES=1 python main.py \
  --model_name_or_path Salesforce/codegen-2B-nl \
  --dataset sstubs \
  --tuning_method ia3 \
  --num_epochs 5 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 6e-5 \
  --lora_r 8 \
  --lora_alpha 16 \
  --do_train



# CUDA_VISIBLE_DEVICES=1 python main.py \
#   --model_name_or_path Salesforce/codegen-6B-multi \
#   --dataset sstubs \
#   --tuning_method ia3 \
#   --num_epochs 5 \
#   --batch_size 1 \
#   --gradient_accumulation_steps 2 \
#   --learning_rate 3e-5 \
#   --lora_r 8 \
#   --lora_alpha 16 \
#   --do_train

# CUDA_VISIBLE_DEVICES=1 python main.py \
#   --model_name_or_path Salesforce/codegen-6B-mono \
#   --dataset sstubs \
#   --tuning_method ia3 \
#   --num_epochs 5 \
#   --batch_size 1 \
#   --gradient_accumulation_steps 2 \
#   --learning_rate 3e-5 \
#   --lora_r 8 \
#   --lora_alpha 16 \
#   --do_train

# CUDA_VISIBLE_DEVICES=1 python main.py \
#   --model_name_or_path Salesforce/codegen-6B-nl \
#   --dataset sstubs \
#   --tuning_method ia3 \
#   --num_epochs 5 \
#   --batch_size 1 \
#   --gradient_accumulation_steps 2 \
#   --learning_rate 3e-5 \
#   --lora_r 8 \
#   --lora_alpha 16 \
#   --do_train