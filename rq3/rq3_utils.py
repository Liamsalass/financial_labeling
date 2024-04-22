import os
import requests
import datasets
import evaluate
import json
import numpy as np
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ======================TOKENIZE AND ALIGN FUNCTIONS======================
# TODO: Combine tokenize and align into a single function?- lots of overlap between functions
def tokenize_and_align_labels_mobilebert(examples):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ
    Function to align tokens and labels. 
    """
    # Load MobileBERT tokenizer.
    tokenizer = return_model_tokenizer("MobileBERT")

    # TODO: Look into max length param for other models
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True
    )

    label_all_tokens = True # Bool which is enabled to label all tokens. Otherwise, first token only.

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_and_align_labels_sec_bert_base(examples):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ
    Function to align tokens and labels for SEC-BERT-BASE
    """
    # Load SEC-BERT-BASE tokenizer.
    tokenizer = return_model_tokenizer("SEC-BERT-BASE")

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    label_all_tokens = True # Bool which is enabled to label all tokens. Otherwise, first token only.

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def tokenize_and_align_labels_sec_bert_num(examples):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ
    Function to align tokens and labels for SEC-BERT-NUM
    """
    # Load SEC-BERT-NUM tokenizer.
    tokenizer = return_model_tokenizer("SEC-BERT-NUM")

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    label_all_tokens = True # Bool which is enabled to label all tokens. Otherwise, first token only.

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_and_align_labels_sec_bert_shape(examples):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ
    Function to align tokens and labels for SEC-BERT-SHAPE
    """
    # Load SEC-BERT-SHAPE tokenizer.
    tokenizer = return_model_tokenizer("SEC-BERT-SHAPE")

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    label_all_tokens = True # Bool which is enabled to label all tokens. Otherwise, first token only.

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
# ======================END OF TOKENIZE AND ALIGN FUNCTIONS======================

# ======================METRICS======================
def compute_metrics(p):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    Function for evaluating model inferences.
    """
    seqeval = evaluate.load("seqeval")
    train_dataset = datasets.load_dataset("nlpaueb/finer-139", split="train")  # Loading train dataset to get tag names.
    finer_tag_names = train_dataset.features["ner_tags"].feature.names

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [finer_tag_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [finer_tag_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Calculate metrics using seqeval
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    # Macro metric calculations
    macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(results)

    return {
        "micro/overall precision": results["overall_precision"],
        "micro/overall recall": results["overall_recall"],
        "micro/overall f1": results["overall_f1"],
        "overall accuracy": results["overall_accuracy"],
        "macro precision": macro_precision,
        "macro recall": macro_recall,
        "macro f1": macro_f1
    }

def calculate_macro_metrics(results):
    """
    Given a Hugging Face seqeval results dictionary, calculate the macro precision, recall, and f1.
    """
    # Keys which do not have individual precision, recall, or f1 scores.
    forbidden_keys = ["overall_precision", "overall_recall", "overall_f1", "overall_accuracy", "total_time_in_seconds", "samples_per_second", "latency_in_seconds"]

    # Get arrays of scores for each metric per token
    individual_precision_scores = [v["precision"] for k, v in results.items() if k not in forbidden_keys]
    individual_recall_scores = [v["recall"] for k, v in results.items() if k not in forbidden_keys]
    individual_f1_scores = [v["f1"] for k, v in results.items() if k not in forbidden_keys]

    # Calculate the macro average, averaging scores across each possible token
    num_features = len(individual_precision_scores)
    macro_precision = sum(individual_precision_scores) / num_features
    macro_recall = sum(individual_recall_scores) / num_features
    macro_f1 = sum(individual_f1_scores) / num_features

    return macro_precision, macro_recall, macro_f1
# ======================END OF METRICS======================

# ======================MODEL + TOKENIZER INSTALLATION FUNCTIONS======================
def get_model_url_and_path(model_name):
    """
    Given a model's name, return the corresponding Hugging Face URL and folder path in the RQ3 directory.
    """
    assert model_name in ["MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"]
    if model_name == "MobileBERT":
        hf_url = "google/mobilebert-uncased"
    elif model_name == "SEC-BERT-BASE":
        hf_url = "nlpaueb/sec-bert-base"
    elif model_name == "SEC-BERT-NUM":
        hf_url = "nlpaueb/sec-bert-num"
    elif model_name == "SEC-BERT-SHAPE":
        hf_url = "nlpaueb/sec-bert-shape"

    folder_path = hf_url.split('/')[1]  # Removes the account name from the HF url, which is where the model is kept
    return hf_url, folder_path

def return_model_tokenizer(model_name):
    """
    Attempts to import the model tokenizer from the Hugging Face Hub.
    If this fails, it loads the tokenizer from the rq3/<model name> folder (after downloading using the check_model_folder function).
    """
    assert model_name in ["MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"]
    hf_url, local_folder_name = get_model_url_and_path(model_name)
    try:
        # Attempts to get tokenizer from HF hub
        tokenizer = AutoTokenizer.from_pretrained(hf_url)
    except OSError:
        # If fails to fetch from HF Hub, look in local rq3/<model name> folder
        check_model_folder(model_name)
        # NOTE: Fails here for SEC-BERT family on Bain machines. Error trace in JSONdecodeerror.txt
        # TODO: Investigate, otherwise SEC-BERT family can't be trained on Bain machines. Will likely have to specify more args in the from_pretrained call? Possibly config?
        tokenizer = AutoTokenizer.from_pretrained(local_folder_name, local_files_only=True)
    
    return tokenizer

def return_model_object(model_name, id2label, label2id, quantization_config):
    """
    Attempts to import the model for token classification from the Hugging Face Hub.
    If this fails, it loads the model from the rq3/<model name> folder (after downloading using the check_model_folder function).
    """
    assert model_name in ["MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"]
    hf_url, local_folder_name = get_model_url_and_path(model_name)

    # 139 labels, but each of them has I- and B-, (along with 0), leading to num_labels=279
    try:
        # Attempts to get model from HF hub
        model = AutoModelForTokenClassification.from_pretrained(
            hf_url, num_labels=279, id2label=id2label, label2id=label2id, quantization_config=quantization_config 
        )
    except OSError:
        # If fails to fetch from Hub, look in local rq3/mobilebert-uncased folder
        check_model_folder(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            local_folder_name, num_labels=279, id2label=id2label, label2id=label2id, local_files_only=True, quantization_config=quantization_config
        )
    
    return model

def check_model_folder(model_name, verbose=False):
    """
    Checks if the folder rq3/<model name> exists and contains all necessary files to load the pretrained model from local files instead of the Hugging Face Hub.
    """
    assert model_name in ["MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"]
    hf_url, local_folder_name = get_model_url_and_path(model_name)

    if model_name == "MobileBERT": # MobileBERT required files
        file_names = ["config.json", ".gitattributes", "pytorch_model.bin", "tokenizer.json", "vocab.txt"]
    else: # SEC-BERT family required files
        file_names = ["config.json", ".gitattributes", "pytorch_model.bin", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]

    if os.path.isdir(local_folder_name): # local folder exists, checking folder contents
        if verbose is True:
            print("The folder " + local_folder_name + " exists.")

        missing_files = []
        # Check if the files exist in the folder
        for file_name in file_names:
            file_path = os.path.join(local_folder_name, file_name)
            if not os.path.isfile(file_path):
                # If missing, keep track of it in an array.
                missing_files.append(file_name)
                if verbose is True:
                    print("File: " + file_name + " is missing.")
        
        # Downloads all missing files
        if len(missing_files) > 0:
            download_model_files(missing_files, local_folder_name, hf_url)
    
    else:  # Folder doesn't exist, installing Hugging Face mobilebert-uncased model contents to this folder now.
        if verbose is True:
            print("The folder " + local_folder_name + " does not exist. Installing all model files to the rq3/" + local_folder_name + " folder.")
        
        # Creating mobilebert-uncased dir and switching to it
        os.mkdir(local_folder_name)
        download_model_files(file_names, local_folder_name, hf_url)

def download_model_files(file_names, model_folder, hf_url):
    """
    Uses the requests directory to download all files from the specified Hugging Face model.
    """
    os.chdir(model_folder)
    for file_name in file_names:
                # Downloading each file to the mobilebert-uncased folder
                try:
                    response = requests.get("https://huggingface.co/" + hf_url + "/resolve/main/" + file_name + "?download=true", timeout=30)
                except requests.exceptions.Timeout:
                    print("Timed out when installing " + file_name + ". Exiting")
                    exit()
                
                open(file_name, "wb").write(response.content)
    
    os.chdir('..')  # Move back one level to return to rq3 as the working directory
# ======================END OF MOBILEBERT INSTALLATION FUNCTIONS======================

def return_peft_config(inference_mode):
    """
    Returns the model's PEFT configuration using QLoRA
    inference_mode determines whether or not the model is being loaded for training(False) or inference(True).
    https://github.com/huggingface/peft/blob/main/examples/token_classification/peft_lora_token_cls.ipynb
    """
    assert inference_mode in [True, False]
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, 
        inference_mode=inference_mode,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="all",  # TODO: Tweak bias in future? lora-only?
        # target_modules=["query", "value"]  #basic lora
        target_modules="all-linear"  # qlora
    )
    return peft_config