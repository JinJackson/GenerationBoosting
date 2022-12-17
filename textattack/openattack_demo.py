import OpenAttack as oa
import transformers
import datasets

model_path = "echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_hidden_states=False)
victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

