
dataset="quora"
# model_type="hfl/chinese-bert-wwm"
model_type="bert-base-uncased"
model_name=${model_type#*/}
batch_size=32
epochs=5
seed=4096
learning_rate='4e-5'

max_length=150
boosting_ratio=0.1
saving_steps=1000



exp_type=boosting_notshuffshell_col1_col2_origin_afterwarmup_newratio
# exp_type=baseline


train_file="../Data/$dataset/clean/train_clean.txt"
# train_file="../Data/$dataset/clean/test_file.txt"
# train_file="../Data/$dataset/clean/augumented_data_file"
dev_file="../Data/$dataset/clean/dev_clean.txt"
test_file="../Data/$dataset/clean/test_clean.txt"


output_dir="/data/zljin/experiments/Paraphrase/Finetune/result/$dataset/$exp_type/$model_name/""bs"$batch_size"_epoch"$epochs"_lr"$learning_rate"_savingsteps"$saving_steps"_seed"$seed"_maxlength"$max_length"_ratio"$boosting_ratio/


echo $train_file
CUDA_VISIBLE_DEVICES=$1 python run_finetune_ratio.py \
--train_file $train_file \
--dataset $dataset \
--dev_file $dev_file \
--test_file $test_file \
--save_dir $output_dir \
--model_type $model_type \
--do_train True \
--do_lower_case True \
--seed $seed \
--learning_rate $learning_rate \
--epochs $epochs \
--batch_size $batch_size \
--max_length $max_length \
--saving_steps $saving_steps \
--gen_device $1 \
--boosting_train \
--boosting_col1 \
--boosting_col2 \
--boosting_origin \
--boosting_ratio $boosting_ratio \
--warmup_steps 0.1

