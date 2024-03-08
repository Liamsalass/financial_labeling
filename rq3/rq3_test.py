import torch
import datasets
from rq3_utils import tokenize_and_align_labels
from transformers import AutoTokenizer, AutoModelForTokenClassification



test_dataset = datasets.load_dataset("nlpaueb/finer-139", split="test")
# NOTE: first 64 samples taken to see that code runs end to end on CPU. Remove line below
test_dataset = test_dataset.select(range(64))

# Tokenize data. NOTE: This is being done with the pretrained model. Check if that's valid/invalid.
tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True)

sentences =  [' '.join(sentence) for sentence in test_dataset["tokens"]]

tokenizer = AutoTokenizer.from_pretrained("rq3_model/checkpoint-8")  # TODO: Get the correct spot for where this path will be
inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)

model = AutoModelForTokenClassification.from_pretrained("rq3_model/checkpoint-8")
with torch.no_grad():
    logits = model(**inputs).logits

predictions = torch.argmax(logits, dim=2)
predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

# TODO: Compare predictions vs actual results