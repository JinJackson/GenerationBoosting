
dataset="LCQMC"
# model_type="hfl/chinese-bert-wwm"
# model_type="nghuyong/ernie-3.0-base-zh"
model_type="hfl/chinese-macbert-large"
# model_type="bert-base-chinese"
model_name=${model_type#*/}
batch_size=32
epochs=3
seed=4096
learning_rate='2e-5'
boosting_method='Gen'
# boosting_method='TextAttack'
# attack_recipe=''
boosting_ratio=1
saving_steps=250    
boarder=20 # bu yong le

exp_type=newboosting_ratiaostop_afterwarmup_nobsboarder
# exp_type=test_textattack


train_file="../Data/$dataset/clean/train_clean.txt"
# train_file="../Data/$dataset/clean/test_file.txt"
# train_file="../Data/$dataset/clean/augumented_data_file"
dev_file="../Data/$dataset/clean/dev_clean.txt"
test_file="../Data/$dataset/clean/test_clean.txt"


output_dir="/data/zljin/experiments/Paraphrase/Finetune/result/$dataset/$exp_type/$model_name/""boosting_method$boosting_method""_boarder$boarder""_bs"$batch_size"_epoch"$epochs"_lr"$learning_rate"_savingsteps"$saving_steps"_seed"$seed"_ratio"$boosting_ratio/


echo $train_file
CUDA_VISIBLE_DEVICES=$1 python run_finetune_ratio_attack.py \
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
--max_length 100 \
--saving_steps $saving_steps \
--gen_device $1 \
--boosting_train \
--boosting_method $boosting_method \
--boosting_col1 \
--boosting_col2 \
--boosting_ratio $boosting_ratio \
--boarder $boarder \
--warmup_steps 0.1
# --boosting_origin \
