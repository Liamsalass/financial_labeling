import os
import requests
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForTokenClassification

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

