import datasets
import evaluate
import numpy as np

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