#!bin/bash

#Gen_type = 'para' or 'nonpara'
Gen_type=$2
source_file=$3
output_dir=$source_file"_"$Gen_type"_generated"



nonpara_model_path=/home/zljin/experiments/Seq2Seq2Generation/src/Finetune/result/LCQMC/midones/nonpara/finetuned-bart-large-chinese/bs8_accumulation4_epoch3_lr2e-5_seed15213
para_model_path=/home/zljin/experiments/Seq2Seq2Generation/src/Finetune/result/LCQMC/midones/para/finetuned-bart-large-chinese/bs8_accumulation4_epoch3_lr2e-5_seed15213

# model_path='choose'

if [ $Gen_type = "para" ]
then
    model_path=$para_model_path
elif [ $Gen_type = 'nonpara' ]
then
    model_path=$nonpara_model_path
fi

echo $Gen_type
echo $model_path
echo $source_file
echo $output_dir

cd ./GenText

CUDA_VISIBLE_DEVICES=$1 python gen_from_file.py \
    --model_type bart \
    --language 'cn' \
    --gen_type $Gen_type \
    --model_name_or_path=$model_path \
    --data_dir $source_file \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --logging_steps 500 \
    --output_dir  $output_dir \
    --num_train_epochs 2 \
    --warmup_steps 500 \
    --save_steps -1 --learning_rate 1e-5 \
    --adam_epsilon 1e-6 \
    --seed 15213 

cd ../
echo 'finished'