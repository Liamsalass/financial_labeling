import datasets
import argparse
import os
import torch
from rq3_utils import tokenize_and_align_labels_mobilebert, compute_metrics, return_mobilebert_tokenizer, return_mobilebert_model
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer


if __name__ == "__main__":
    print("CUDA available: ", torch.cuda.is_available())
    print("CUDA current device: ", torch.cuda.current_device())  # CPU is -1. Else GPU
    # TODO: How to verify the use of device in the Trainer call in HF?
    # https://discuss.huggingface.co/t/setting-specific-device-for-trainer/784/19
    
    # Load the train and val dataset splits.
    # TODO: Add progress bar/update for dataset loading?
    train_dataset = datasets.load_dataset("nlpaueb/finer-139", split="train")
    val_dataset = datasets.load_dataset("nlpaueb/finer-139", split="validation")

    # Parsing command line args
    # TODO: Tune hyperparameters, defaults here are left from the Hugging Face tutorial
    parser = argparse.ArgumentParser(description='CMPE 351 RQ3 Training code')
    parser.add_argument('-subset', type=int, default=-1, help='Specify to use a subset of the train and val set. If left empty, use the entire train and val sets.')
    parser.add_argument('-output_checkpoint_path', type=str, default="rq3_model", help='Specify the relative path to the checkpoint folder.')
    parser.add_argument('-lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('-train_batch_size', type=int, default=16, help='Train batch size per device')
    parser.add_argument('-val_batch_size', type=int, default=16, help='Val batch size per device')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-weight_decay', type=float, default=0.01)
    arguments = parser.parse_args()
    
    subset_size = arguments.subset
    checkpoint_path = arguments.output_checkpoint_path
    learning_rate = arguments.lr
    train_batch_size_per_device = arguments.train_batch_size
    val_batch_size_per_device = arguments.val_batch_size
    epochs = arguments.epochs
    weight_decay = arguments.weight_decay

    # Verifying command line args
    if subset_size != -1:
        assert 0 < subset_size < len(val_dataset)
        # Selects the specified # of samples from the subset argument.
        train_dataset = train_dataset.select(range(subset_size))
        val_dataset = val_dataset.select(range(subset_size))
        print("Training MobileBERT on a subset of FiNER-139 train/val sets with " + str(subset_size) + " samples each.")
    else:
        print("Training MobileBERT on full FiNER-139 train/val sets with " + str(len(train_dataset)) + " train samples and " + str(len(val_dataset)) + " val samples.")
    
    full_checkpoint_path = os.getcwd() + "/" + checkpoint_path
    assert os.path.isdir(full_checkpoint_path) is True  # Verify checkpoint dir exists
    print("Training MobileBERT checkpoints will be stored in the " + checkpoint_path + " folder")

    assert 0 < learning_rate < 1
    assert 1 <= train_batch_size_per_device <= len(train_dataset)
    assert 1 <= val_batch_size_per_device <= len(val_dataset)
    assert 1 <= epochs <= 10  # MobileBERT paper explains that they fine tune with 10 epochs max in section 4.4.2.
    assert 0 < weight_decay < 1

    # Getting array of tags/labels
    finer_tag_names = train_dataset.features["ner_tags"].feature.names

    # Load MobileBERT tokenizer.
    tokenizer = return_mobilebert_tokenizer()

    # https://stackoverflow.com/questions/64320883/the-size-of-tensor-a-707-must-match-the-size-of-tensor-b-512-at-non-singleto

    # Tokenize each section of the dataset.
    tokenized_train = train_dataset.map(tokenize_and_align_labels_mobilebert, batched=True)
    tokenized_val = val_dataset.map(tokenize_and_align_labels_mobilebert, batched=True)

    # For creating batches of examples
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # id2label and label2id dictionaries for loading the model.
    id2label = {i: element for i, element in enumerate(finer_tag_names)}
    label2id = {value: i for i, value in enumerate(finer_tag_names)}

    # Uses the function in rq3_utils to get the pretrained MobileBERT model.
    model = return_mobilebert_model(id2label, label2id)

    print("MobileBERT Parameter Count: ", model.num_parameters())
    # TODO: Include support to use SEC-BERT variants for training? Want to compare the efficiency and performance of both. 
    # Maybe distillbert also: https://huggingface.co/dslim/distilbert-NER#:~:text=distilbert%2DNER%20is%20the%20fine,Named%20Entity%20Recognition%20(NER).

    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size_per_device,
        per_device_eval_batch_size=val_batch_size_per_device,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    # NOTE: https://huggingface.co/learn/nlp-course/en/chapter7/2 , custom training loop?

    # Model Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    print("\n\n-----TRAINING-----")
    trainer.train()
