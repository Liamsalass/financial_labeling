import argparse
import os
import datasets
import torch
import torch_optimizer
import wandb
from copy import deepcopy
from rq3_utils import compute_metrics
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification, AutoTokenizer, TrainerCallback, BitsAndBytesConfig

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

    # Parsing command line args
    parser = argparse.ArgumentParser(description='CMPE 351 RQ3 Training code')
    parser.add_argument('-model_name', type=str, default='MobileBERT', help='Selected model to train. Enter one of "MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"')
    parser.add_argument('-subset', type=int, default=-1, help='Specify to use a subset of the train and val set. If left empty, use the entire train and val sets.')
    parser.add_argument('-output_checkpoint_path', type=str, default="rq3_mobilebert_model", help='Specify the relative path to the output checkpoint folder.')
    parser.add_argument('-resume_from_checkpoint_path', type=str, default=None, help='Specify the checkpoint to continue training from. Does not train from checkpoint by default.')
    parser.add_argument('-lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('-train_batch_size', type=int, default=16, help='Train batch size per device')
    parser.add_argument('-val_batch_size', type=int, default=16, help='Val batch size per device')
    parser.add_argument('-epochs', type=int, default=2, help="If continuing training from a checkpoint, enter the total number of epochs you want to reach (start epoch + desired # of epochs)")
    parser.add_argument('-peft', type=int, default=1, help='Specify whether or not to use PEFT during training [0/1].')
    parser.add_argument('-quantize', type=int, default=0, help='Specify whether or not to use FP4 quantization during training [0/1].')
    arguments = parser.parse_args()

    # Command line args into variables
    model_name = arguments.model_name
    subset_size = arguments.subset
    checkpoint_path = arguments.output_checkpoint_path
    resume_from_checkpoint_path = arguments.resume_from_checkpoint_path
    learning_rate = arguments.lr
    train_batch_size_per_device = arguments.train_batch_size
    val_batch_size_per_device = arguments.val_batch_size
    epochs = arguments.epochs
    use_peft = arguments.peft
    quantize = arguments.quantize

    print("CUDA available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA current device: ", torch.cuda.current_device())  # CPU is -1. Else GPU
        # FP4 requires GPU, setting model quantization config accordingly
        if use_peft == 1 and quantize == 1:
            # NOTE: Quantization is causing some very odd behaviour with NaN loss and high 90s accuracy. Not working currently
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
    else:
        print("CUDA unavailable, using CPU")
        bnb_config = None
    
    if bnb_config is not None:
        print("Performing FP4 quantization during training.")
    
    # Load the train and val dataset splits.
    print("Loading train dataset.")
    train_dataset = datasets.load_dataset("nlpaueb/finer-139", split="train")
    print("Train dataset loaded. Loading val dataset.")
    val_dataset = datasets.load_dataset("nlpaueb/finer-139", split="validation")
    print("Val dataset loaded")

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

    if resume_from_checkpoint_path is not None:
        assert os.path.isdir(resume_from_checkpoint_path)
        print("Training " + model_name + " starting from the " + resume_from_checkpoint_path + " checkpoint")

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

    # Load the tokenizer and model object
    if resume_from_checkpoint_path is not None:  # Load model and tokenizer from starting_checkpoint_path
        if use_peft == 1:
            from peft import PeftModel, PeftConfig
            config = PeftConfig.from_pretrained(resume_from_checkpoint_path)
            inference_model = AutoModelForTokenClassification.from_pretrained(
            config.base_model_name_or_path, num_labels=279, id2label=id2label, label2id=label2id)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(inference_model, resume_from_checkpoint_path)
        else:
            model = AutoModelForTokenClassification.from_pretrained(resume_from_checkpoint_path)
            tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint_path)
    else:  # Load model from Hugging Face
        if model_name == "MobileBERT": # Load MobileBERT
            from rq3_utils import return_mobilebert_model, return_mobilebert_tokenizer, tokenize_and_align_labels_mobilebert
            model = return_mobilebert_model(id2label, label2id, bnb_config)
            tokenizer = return_mobilebert_tokenizer()
        else:  # Load model from SEC-BERT family
            if model_name == "SEC-BERT-BASE":
                sec_bert_url = "nlpaueb/sec-bert-base"
            elif model_name == "SEC-BERT-NUM":
                sec_bert_url = "nlpaueb/sec-bert-num"
            elif model_name == "SEC-BERT-SHAPE":
                sec_bert_url = "nlpaueb/sec-bert-shape"
            model = AutoModelForTokenClassification.from_pretrained(sec_bert_url, num_labels=279, id2label=id2label, label2id=label2id, quantization_config=bnb_config)
            tokenizer = AutoTokenizer.from_pretrained(sec_bert_url)
    
        # PEFT
        if use_peft == 1:
            from rq3_utils import return_peft_config
            from peft import get_peft_model
            peft_config = return_peft_config(inference_mode=False)
            model = get_peft_model(model, peft_config) 
        else:  # Only train the output/classification layer, freeze all other gradients
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
    
    if use_peft == 1:
            print(model_name + " PEFT parameter overview: ")
            model.print_trainable_parameters()
    else:
        print(model_name, "Total Parameter Count: ", model.num_parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model_name, "Trainable Parameter Count: ", str(trainable_params))


    # Tokenize the train and val sets
    if model_name == "MobileBERT":
        from rq3_utils import tokenize_and_align_labels_mobilebert
        tokenize_and_align_fn = tokenize_and_align_labels_mobilebert
    elif model_name == "SEC-BERT-BASE":
        from rq3_utils import tokenize_and_align_labels_sec_bert_base
        tokenize_and_align_fn = tokenize_and_align_labels_sec_bert_base
    elif model_name == "SEC-BERT-NUM":
        from rq3_utils import tokenize_and_align_labels_sec_bert_num
        tokenize_and_align_fn = tokenize_and_align_labels_sec_bert_num
    elif model_name == "SEC-BERT-SHAPE":
        from rq3_utils import tokenize_and_align_labels_sec_bert_shape
        tokenize_and_align_fn = tokenize_and_align_labels_sec_bert_shape
    
    tokenized_train = train_dataset.map(tokenize_and_align_fn, batched=True)
    tokenized_val = val_dataset.map(tokenize_and_align_fn, batched=True)

    # For creating batches of examples
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Optimizer for each model.
    if model_name == "MobileBERT":
        optimizer = torch_optimizer.Lamb(model.parameters(), lr=learning_rate)
    else:  # SEC-BERT family
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_with_checkpoint = False
    if resume_from_checkpoint_path is not None:
        training_with_checkpoint = True
        # NOTE: automatically takes last checkpoint to train from.

    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        per_device_train_batch_size=train_batch_size_per_device,
        per_device_eval_batch_size=val_batch_size_per_device,
        lr_scheduler_type="constant",  # Disables LR scheduling which happens by default in HF
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        use_cpu=False
    )

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
    trainer.train(resume_from_checkpoint=training_with_checkpoint)