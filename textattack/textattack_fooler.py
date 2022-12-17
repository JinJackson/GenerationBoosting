import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-QQP')
tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-QQP')

# model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese')
# tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
#dyyyyyyyy/LCQMC_BERT-base-Chinese 


import textattack
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult

print('1')
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
#什么牌子口红好用不贵	什么牌子的口红好用又便宜	1
data = [(("The man is sleeping on the bed", "The man has fallen asleep on the bed"), 1), (("The man has fallen asleep on the bed", "The man is sleeping on the bed"), 1)]
dataset = textattack.datasets.Dataset(data, input_columns=['text1', 'text2'])

print('2')
#PWWSRen2019
#TextBuggerLi2018
#BERTAttackLi2020
#BAEGarg2019
#CheckList2020
attack = textattack.attack_recipes.BERTAttackLi2020.build(model_wrapper)
attack_args1 = textattack.AttackArgs(num_examples=len(dataset), log_to_csv="log1.csv", random_seed=765, checkpoint_interval=5, checkpoint_dir="checkpoints", disable_stdout=True, log_to_txt='log1.txt')
attack_args2 = textattack.AttackArgs(num_examples=len(dataset), log_to_csv="log2.csv", random_seed=1024, checkpoint_interval=5, checkpoint_dir="checkpoints", disable_stdout=True, log_to_txt='log2.txt')
print('3')



import time
print('start')
start_time = time.time()
attacker1 = textattack.Attacker(attack, dataset, attack_args1)
res1 = attacker1.attack_dataset()

attacker2 = textattack.Attacker(attack, dataset, attack_args2)
res2 = attacker2.attack_dataset()
# import pdb; pdb.set_trace()
end_time = time.time()

print(end_time - start_time)

# import pdb; pdb.set_trace()
print(res1[0].original_text(), res1[0].perturbed_text())

#'Text1: A man is sleeping on the bed.\nText2: The man is almost sleeping.'

# print(res2[0].original_text(), res2[0].perturbed_text())
# if isinstance(res1[0], SuccessfulAttackResult):
#     perturbed_text=res[0].perturbed_text()
#     original_text = res[0].original_text()
#     print(perturbed_text)
#     print(original_text)

# elif isinstance(res1[0], FailedAttackResult):
#     perturbed_text=res[0].perturbed_text()
#     original_text = res[0].original_text()
#     print(perturbed_text)
#     print(original_text)