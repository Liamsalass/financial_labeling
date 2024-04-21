import datasets
import argparse
import os
import numpy as np
import torch
import torch_optimizer
import wandb
from peft import get_peft_model
from utils.metrics import compute_metrics
from utils.tokenize_and_align import tokenize_and_align_labels_mobilebert, tokenize_and_align_labels_sec_bert_base, tokenize_and_align_labels_sec_bert_num, tokenize_and_align_labels_sec_bert_shape
from utils.rq3_utils import return_mobilebert_tokenizer, return_mobilebert_model, return_peft_config
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification, AutoTokenizer
    

if __name__ == "__main__":
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
    parser.add_argument('-model_name', type=str, default='MobileBERT', help='Selected model to test. Enter one of "MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"')
    parser.add_argument('-subset', type=int, default=-1, help='Specify to use a subset of the train and val set. If left empty, use the entire train and val sets.')
    parser.add_argument('-wandb_project_path', type=str, default="rq3_mobilebert_sweep", help='Specify the relative path to the wandb project folder.')
    parser.add_argument('-peft', type=int, default=1, help='Specify whether or not to use PEFT during training [0/1].')
    arguments = parser.parse_args()
    
    # Command line args into variables
    model_name = arguments.model_name
    subset_size = arguments.subset
    wandb_path = arguments.wandb_project_path
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
    
    full_wandb_path = os.getcwd() + "/" + wandb_path
    if os.path.isdir(full_wandb_path) is False:
        os.mkdir(wandb_path)
    assert os.path.isdir(wandb_path)
    print("wandb sweep for " + model_name + " will be stored in the " + wandb_path + " project folder")

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
        optimizer = torch_optimizer.Lamb(model.parameters())
    else:  # SEC-BERT family
        optimizer = torch.optim.Adam(model.parameters())

    print("\n\n-----PERFORMING HYPERPARAMETER SEARCH-----")

    if model_name == "MobileBERT":
        # Values taken from MobileBERT paper fine-tuning section: https://arxiv.org/pdf/2004.02984.pdf
        wandb_config = {
            "method": "grid",  # TODO: Consider random method, grid may be very costly
            "parameters": {
                "epochs": {"values": (np.arange(start=2, stop=11, step=1)).tolist()}, # 2-10 inclusive
                "learning_rate": {"values": (np.arange(start=1e-5, stop=11e-5, step=1e-5).tolist())},  # 1e-5 to 10e-5 inclusive
                "per_device_train_batch_size": {"values": [16, 32, 48]}
            },
        }
    else:  # SEC-BERT family
        # Values taken from BERT paper fine-tuning section: https://arxiv.org/pdf/1810.04805.pdf
        wandb_config = {
            "method": "grid", # TODO: Consider random method, grid may be very costly
            "parameters": {
                "learning_rate": {"values": [5e-5, 3e-5, 2e-5]},
                "per_device_train_batch_size": {"values": [16, 32]},
                "epochs": {"values": [2, 3, 4]}
            },
        }

    # Weights and biases login + setup
    # TODO: Set up configuration with wandb login- 
    wandb.login()
    # Create 2 environment variables
    os.environ["WANDB_PROJECT"] = wandb_path
    os.environ["WANDB_LOG_MODEL"] = "true"

    sweep_id = wandb.sweep(wandb_config, project=wandb_path)

    def train(config=None):
        # code staring point: https://wandb.ai/matt24/vit-snacks-sweeps/reports/Hyperparameter-Search-for-HuggingFace-Transformer-Models--VmlldzoyMTUxNTg0
        with wandb.init(config=config):
            # set sweep configuration
            config = wandb.config

            # set training arguments
            training_args = TrainingArguments(
                output_dir=wandb_path,
                report_to='wandb',  # Turn on Weights & Biases logging
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.per_device_train_batch_size,
                per_device_eval_batch_size=16,
                save_strategy='epoch',
                evaluation_strategy='epoch',
                logging_strategy='epoch',
                load_best_model_at_end=True,
            )

            # define training loop
            trainer = Trainer(
                model=model,
                args=training_args,
                optimizers=(optimizer, None),  # TODO: scheduler? Currently have no scheduler. Look into papers to see what this should be.
                data_collator=data_collator,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                compute_metrics=compute_metrics  # TODO: Compute metrics for SEC-BERT family?
            )

            # start training loop
            trainer.train()
    
    wandb.agent(sweep_id, train)