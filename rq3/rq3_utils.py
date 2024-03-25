import datasets
import evaluate
import numpy as np
import re
import spacy
from transformers import AutoTokenizer

def tokenize_and_align_labels_mobilebert(examples):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    Function to align tokens and labels. 
    """
    # TODO: Change the tokenizer which is used when testing?
    # TODO: Fast tokenizer?
    # Load MobileBERT tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    Function for evaluating model inferences.
    """
    seqeval = evaluate.load("seqeval")
    train_dataset = datasets.load_dataset("nlpaueb/finer-139", split="test")  # Loading train dataset to get tag names.
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

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def sec_bert_num_preprocess(examples):
    """
    From https://huggingface.co/nlpaueb/sec-bert-num
    """
    # TODO: Can we make this parallelizable somehow? Seems slow. Same for SEC-BERT-SHAPE
    spacy_tokenizer = spacy.load("en_core_web_sm")

    for idx, sentence in enumerate(examples["tokens"]):
        tokens = [spacy_tokenizer(tok).text for tok in sentence]
        processed_text = []
        for token in tokens:
            if re.fullmatch(r"(\d+[\d,.]*)|([,.]\d+)", token):
                processed_text.append('[NUM]')
            else:
                processed_text.append(token)
        examples["tokens"][idx] = processed_text
    return examples


# NOTE: spacy needs to install en_core_web_sm through Python: python -m spacy download en_core_web_sm
# TODO: Update dependencies or setup?
def sec_bert_shape_preprocess(examples):
    """
    From: https://huggingface.co/nlpaueb/sec-bert-shape
    """
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-shape")
    
    spacy_tokenizer = spacy.load("en_core_web_sm")

    for idx, sentence in enumerate(examples["tokens"]):
        tokens = [spacy_tokenizer(tok).text for tok in sentence]
        processed_text = []
        for token in tokens:
            if re.fullmatch(r"(\d+[\d,.]*)|([,.]\d+)", token):
                shape = '[' + re.sub(r'\d', 'X', token) + ']'
                if shape in tokenizer.additional_special_tokens:
                    processed_text.append(shape)
                else:
                    processed_text.append('[NUM]')
            else:
                processed_text.append(token)
        examples["tokens"][idx] = processed_text
    return examples