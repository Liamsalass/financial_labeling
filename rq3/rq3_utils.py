import datasets
import evaluate
import numpy as np
from transformers import AutoTokenizer

def tokenize_and_align_labels_mobilebert(examples):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    Function to align tokens and labels. 
    """
    # Load MobileBERT tokenizer.
    # TODO: Change the tokenizer which is used when testing?
    # TODO: Fast tokenizer?
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