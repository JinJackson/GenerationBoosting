import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-QQP')
tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-QQP')

import textattack
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult

print('1')
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

data = [(("A man is sleeping on the bed.", "The man is almost sleeping."), 1), (("The man is almost sleeping.", "A man is sleeping on the bed."), 0)]
dataset = textattack.datasets.Dataset(data, input_columns=['text1', 'text2'])

print('2')

attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
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

print(res1[0].original_text(), res1[0].perturbed_text())
print(res2[0].original_text(), res2[0].perturbed_text())
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