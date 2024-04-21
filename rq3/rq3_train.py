import argparse
import os
import datasets
import torch
import torch_optimizer
import wandb
from copy import deepcopy
from peft import get_peft_model
from utils.rq3_utils import compute_metrics, return_mobilebert_tokenizer, return_mobilebert_model, return_peft_config
from utils.tokenize_and_align import tokenize_and_align_labels_mobilebert, tokenize_and_align_labels_sec_bert_base, tokenize_and_align_labels_sec_bert_num, tokenize_and_align_labels_sec_bert_shape
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification, AutoTokenizer, TrainerCallback


class CustomCallback(TrainerCallback):
    """
    This custom callback ensures that training metrics are included in the Hugging Face trainer_state.json logs.
    From: https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/7
    """
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


if __name__ == "__main__":
    wandb.init(mode="disabled")  # Disable wandb for this file.

    print("CUDA available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA current device: ", torch.cuda.current_device())  # CPU is -1. Else GPU
    else:
        print("CUDA unavailable, using CPU")
    
    # Load the train and val dataset splits.
    train_dataset = datasets.load_dataset("nlpaueb/finer-139", split="train")
    print("Train dataset loaded")
    val_dataset = datasets.load_dataset("nlpaueb/finer-139", split="validation")
    print("Val dataset loaded")

    # Parsing command line args
    parser = argparse.ArgumentParser(description='CMPE 351 RQ3 Training code')
    parser.add_argument('-model_name', type=str, default='MobileBERT', help='Selected model to train. Enter one of "MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"')
    parser.add_argument('-subset', type=int, default=-1, help='Specify to use a subset of the train and val set. If left empty, use the entire train and val sets.')
    parser.add_argument('-output_checkpoint_path', type=str, default="rq3_mobilebert_model", help='Specify the relative path to the checkpoint folder.')
    parser.add_argument('-lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('-train_batch_size', type=int, default=16, help='Train batch size per device')
    parser.add_argument('-val_batch_size', type=int, default=16, help='Val batch size per device')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-peft', type=int, default=1, help='Specify whether or not to use PEFT during training [0/1].')
    arguments = parser.parse_args()

    # Command line args into variables
    model_name = arguments.model_name
    subset_size = arguments.subset
    checkpoint_path = arguments.output_checkpoint_path
    learning_rate = arguments.lr
    train_batch_size_per_device = arguments.train_batch_size
    val_batch_size_per_device = arguments.val_batch_size
    epochs = arguments.epochs
    use_peft = arguments.peft

    # Verifying command line args
    assert model_name in ["MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"]

    if subset_size != -1:
        assert 0 < subset_size < len(val_dataset)
        # Selects the specified # of samples from the subset argument.
        train_dataset = train_dataset.select(range(subset_size))
        val_dataset = val_dataset.select(range(subset_size))
        print("Training " + model_name + " on a subset of FiNER-139 train/val sets with " + str(subset_size) + " samples each.")
    else:
        print("Training " + model_name + " on full FiNER-139 train/val sets with " + str(len(train_dataset)) + " train samples and " + str(len(val_dataset)) + " val samples.")
    
    full_checkpoint_path = os.getcwd() + "/" + checkpoint_path
    if os.path.isdir(full_checkpoint_path) is False:
        os.mkdir(checkpoint_path)
    assert os.path.isdir(checkpoint_path)
    print("Training " + model_name + " checkpoints will be stored in the " + checkpoint_path + " folder")

    assert 0 < learning_rate < 1
    assert 1 <= train_batch_size_per_device <= len(train_dataset)
    assert 1 <= val_batch_size_per_device <= len(val_dataset)
    assert 1 <= epochs <= 10  # MobileBERT paper explains that they fine tune with 10 epochs max in section 4.4.2.
    assert use_peft in [0, 1]

    # Getting array of tags/labels
    finer_tag_names = train_dataset.features["ner_tags"].feature.names

    # id2label and label2id dictionaries for loading the model.
    id2label = {i: element for i, element in enumerate(finer_tag_names)}
    label2id = {value: i for i, value in enumerate(finer_tag_names)}

    # Load the tokenizer and model object, then tokenize the train and val sets
    if model_name == "MobileBERT":
        model = return_mobilebert_model(id2label, label2id)
        tokenizer = return_mobilebert_tokenizer()
        tokenized_train = train_dataset.map(tokenize_and_align_labels_mobilebert, batched=True)
        tokenized_val = val_dataset.map(tokenize_and_align_labels_mobilebert, batched=True)
    elif model_name == "SEC-BERT-BASE":
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-base", num_labels=279, id2label=id2label, label2id=label2id)
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")
        tokenized_train = train_dataset.map(tokenize_and_align_labels_sec_bert_base, batched=True)
        tokenized_val = val_dataset.map(tokenize_and_align_labels_sec_bert_base, batched=True)
    elif model_name == "SEC-BERT-NUM":
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-num", num_labels=279, id2label=id2label, label2id=label2id)
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-num")
        tokenized_train = train_dataset.map(tokenize_and_align_labels_sec_bert_num, batched=True) # Apply SEC-BERT-NUM preprocessing
        tokenized_val = val_dataset.map(tokenize_and_align_labels_sec_bert_num, batched=True)
    elif model_name == "SEC-BERT-SHAPE":
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-shape", num_labels=279, id2label=id2label, label2id=label2id)
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-shape")
        tokenized_train = train_dataset.map(tokenize_and_align_labels_sec_bert_shape, batched=True) # Apply SEC-BERT-SHAPE preprocessing
        tokenized_val = val_dataset.map(tokenize_and_align_labels_sec_bert_shape, batched=True)

    # NOTE: SEC-BERT family of models is assumed to have a linear classification layer after BERT. Unsure if there should be something else.
        # Check FiNER-139 paper for and GitHub for true architecture.

    # For creating batches of examples
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # PEFT
    if use_peft == 1:
        peft_config = return_peft_config(inference_mode=False, model_name=model_name)
        model = get_peft_model(model, peft_config) 
        print(model_name + " PEFT parameter overview: ")
        model.print_trainable_parameters()
    else:  # Only train the output/classification layer, freeze all other gradients
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print(model_name, "Total Parameter Count: ", model.num_parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model_name, "Trainable Parameter Count: ", str(trainable_params))

    # Optimizer for each model.
    if model_name == "MobileBERT":
        optimizer = torch_optimizer.Lamb(model.parameters(), lr=learning_rate)
    else:  # SEC-BERT family
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        per_device_train_batch_size=train_batch_size_per_device,
        per_device_eval_batch_size=val_batch_size_per_device,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        use_cpu=False
    )

    # NOTE: https://huggingface.co/learn/nlp-course/en/chapter7/2 , custom training loop?

    # Model Trainer object
    trainer = Trainer(
        model=model,
        optimizers=[optimizer, None],
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Add callback to track training metrics
    trainer.add_callback(CustomCallback(trainer))

    print("\n\n-----TRAINING-----")
    trainer.train()