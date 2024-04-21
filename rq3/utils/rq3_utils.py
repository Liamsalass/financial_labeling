import datasets
import evaluate
import numpy as np
import os
import requests
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForTokenClassification

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


def return_mobilebert_tokenizer():
    """
    Attempts to import the MobileBERT tokenizer from the Hugging Face Hub. (https://huggingface.co/google/mobilebert-uncased)
    If this fails, it loads the MobileBERT tokenizer from the rq3/mobilebert-uncased folder (after downloading using the check_mobilebert_folder function).
    """
    try:
        # Attempts to get tokenizer from HF hub
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    except OSError:
        # If fails to fetch from Hub, look in local rq3/mobilebert-uncased folder
        check_mobilebert_folder()
        tokenizer = AutoTokenizer.from_pretrained("mobilebert-uncased", local_files_only=True)
    
    return tokenizer

def return_mobilebert_model(id2label, label2id):
    """
    Attempts to import the MobileBERT model for token classification from the Hugging Face Hub. (https://huggingface.co/google/mobilebert-uncased)
    If this fails, it loads the MobileBERT model from the rq3/mobilebert-uncased folder (after downloading using the check_mobilebert_folder function).
    """
    # 139 labels, but each of them has I- and B-, (along with 0), leading to num_labels=279
    try:
        # Attempts to get model from HF hub
        model = AutoModelForTokenClassification.from_pretrained(
            "google/mobilebert-uncased", num_labels=279, id2label=id2label, label2id=label2id
        )
    except OSError:
        # If fails to fetch from Hub, look in local rq3/mobilebert-uncased folder
        check_mobilebert_folder()
        model = AutoModelForTokenClassification.from_pretrained(
            "mobilebert-uncased", num_labels=279, id2label=id2label, label2id=label2id, local_files_only=True
        )
    
    return model

def return_peft_config(inference_mode, model_name):
    """
    Returns the PEFT configuration for the corresponding model architecture.
    inference_mode determines whether or not the model is being loaded for training(False) or inference(True).
    https://github.com/huggingface/peft/blob/main/examples/token_classification/peft_lora_token_cls.ipynb
    """
    assert inference_mode in [True, False]
    assert model_name in ["MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"]
    
    # Set the prefix in the HF model object, since the lora elements are the same for all 4
    if model_name == "MobileBERT":
        prefix = "mobilebert"
    else:
        prefix = "BertModel"
    
    lora_modules = [ prefix + ".embeddings.word_embeddings", prefix + ".embeddings.position_embeddings",
                    prefix + ".embeddings.token_type_embeddings", prefix + ".embeddings.embedding_transformation"]

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, inference_mode=inference_mode, r=16, lora_alpha=16, lora_dropout=0.1, bias="all", target_modules=lora_modules
    )
    # TODO: Investigate Lora parameters, used ones from tutorial example.
    return peft_config

def check_mobilebert_folder(verbose=False):
    """
    Checks if the folder rq3/mobilebert-uncased exists and contains all necessary files to load the pretrained model from local files instead of the Hugging Face Hub.
    """
    folder_path = "mobilebert-uncased"
    file_names = ["config.json", "gitattributes", "pytorch_model.bin", "tokenizer.json", "vocab.txt"]

    if os.path.isdir(folder_path):  # mobilebert-uncased exists, checking folder contents
        if verbose is True:
            print("The folder " + folder_path + " exists.")

        missing_files = []
        # Check if the files exist in the folder
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.isfile(file_path):
                # If missing, keep track of it in an array.
                missing_files.append(file_name)
                if verbose is True:
                    print("File: " + file_name + " is missing.")
        
        # Downloads all missing files
        if len(missing_files) > 0:
            download_mobilebert_files(missing_files)
    
    else:  # Folder doesn't exist, installing Hugging Face mobilebert-uncased model contents to this folder now.
        if verbose is True:
            print("The folder " + folder_path + " does not exist. Installing all MobileBERT files to the rq3/mobilebert-uncased folder.")
        
        # Creating mobilebert-uncased dir and switching to it
        os.mkdir('mobilebert-uncased')
        download_mobilebert_files(file_names)


def download_mobilebert_files(file_names):
    """
    Uses the requests directory to download all files from the mobilebert-uncased model from Hugging Face.
    """
    os.chdir('mobilebert-uncased')
    for file_name in file_names:
                # Downloading each file to the mobilebert-uncased folder
                try:
                    response = requests.get("https://huggingface.co/google/mobilebert-uncased/resolve/main/" + file_name + "?download=true", timeout=30)
                except requests.exceptions.Timeout:
                    print("Timed out when installing " + file_name + ". Exiting")
                    exit()
                
                open(file_name, "wb").write(response.content)
    
    os.chdir('..')  # Move back one level to return to rq3 as the working directory

