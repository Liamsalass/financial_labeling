import datasets
from rq3_utils import tokenize_and_align_labels, compute_metrics
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer


# Load the train, val, and test dataset splits.
train_dataset = datasets.load_dataset("nlpaueb/finer-139", split="train")
val_dataset = datasets.load_dataset("nlpaueb/finer-139", split="validation")

# NOTE: first 64 samples taken to see that code runs end to end on CPU. Remove below 2 lines.
train_dataset = train_dataset.select(range(64))
val_dataset = val_dataset.select(range(64))


# Getting array of tags/labels
finer_tag_names = train_dataset.features["ner_tags"].feature.names

# Load MobileBERT tokenizer.
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# Tokenize each section of the dataset.
tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True)

# For creating batches of examples
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# id2label and label2id dictionaries for loading the model.
id2label = {i: element for i, element in enumerate(finer_tag_names)}
label2id = {value: i for i, value in enumerate(finer_tag_names)}

# Importing MobileBERT model
model = AutoModelForTokenClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=279, id2label=id2label, label2id=label2id
)
# NOTE: 139 labels, but each of them has I- and B-, (along with 0), leading to num_labels=279

# Training arguments
training_args = TrainingArguments(
    output_dir="rq3_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Model Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
