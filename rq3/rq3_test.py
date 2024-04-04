import datasets
import argparse
import os
import torch
from evaluate import evaluator
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
from rq3_utils import sec_bert_num_preprocess, sec_bert_shape_preprocess

# TODO: Device support for GPU runs.
if __name__ == "__main__":
    print("CUDA available: ", torch.cuda.is_available())
    print("CUDA current device: ", torch.cuda.current_device())  # CPU is -1. Else GPU
    # TODO: Verify device usage?

    # Loading the test dataset. NOTE: In future, can possibly specify path to this dataset with command line args. Not needed right now.
    test_dataset = datasets.load_dataset("nlpaueb/finer-139", split="test")

    # Parsing command line args
    parser = argparse.ArgumentParser(description='CMPE 351 RQ3 Testing code')
    parser.add_argument('-model_name', type=str, default='MobileBERT', help='Selected model to test. Enter one of "MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"')
    parser.add_argument('-subset', type=int, default=-1, help='Specify to use a subset of the test set. If left empty, use the entire test set.')
    parser.add_argument('-checkpoint_path', type=str, default="rq3_model/checkpoint-32", help='Specify the relative path to the Hugging Face model checkpoint to evaluate.')
    arguments = parser.parse_args()
    
    model_name = arguments.model_name
    subset_size = arguments.subset
    checkpoint_path = arguments.checkpoint_path

    # Verifying command line args
    assert model_name in ["MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"]

    if subset_size != -1:
        assert 0 < subset_size < len(test_dataset)
        test_dataset = test_dataset.select(range(subset_size))  # Selects the specified # of samples from the subset argument.
        print("Testing " + model_name + " on subset of FiNER-139 test set with " + str(subset_size) + " samples.")
    else:
        print("Testing " + model_name + " on full FiNER-139 test set with " + str(len(test_dataset)) + " samples.")
    
    if model_name == "MobileBERT":
        # NOTE: Check Below doesn't check for the contents of the directory. Could possibly verify this to ensure that the weights are in the folder.
        # NOTE: Checkpoint paths are only being used for MobileBERT. May need to use them for the SEC-BERT models since these don't come with classifier weights.
        full_checkpoint_path = os.getcwd() + "/" + checkpoint_path
        assert os.path.isdir(full_checkpoint_path) is True  # Verify checkpoint dir exists
        print("Testing " + model_name + " checkpoint stored in the " + checkpoint_path + " folder")

    if model_name == "MobileBERT":
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    elif model_name == "SEC-BERT-BASE":
        # NOTE: SEC-BERT Classifier weights and biases are not loaded. The HF pretrained model is meant for fill-masking. See if we can load these weights from somewhere?
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-base")
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")
    elif model_name == "SEC-BERT-NUM":
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-num")
        test_dataset = test_dataset.map(sec_bert_num_preprocess, batched=True)  # Apply SEC-BERT-NUM preprocessing
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-num")
    elif model_name == "SEC-BERT-SHAPE":
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-shape")
        test_dataset = test_dataset.map(sec_bert_shape_preprocess, batched=True)  # Apply SEC-BERT-SHAPE preprocessing
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-shape")
    
    print(model_name + " Parameter Count: ", model.num_parameters())

    # TODO: Batch this?
    # Possible Batching link: https://huggingface.co/docs/transformers/en/main_classes/pipelines
    # https://stackoverflow.com/questions/75932605/getting-the-input-text-from-transformers-pipeline
    classifier_pipeline = pipeline(task="token-classification", model=model, tokenizer=tokenizer)
    task_evaluator = evaluator("token-classification")
    test_results = task_evaluator.compute(model_or_pipeline=classifier_pipeline, data=test_dataset, metric="seqeval")
    # TODO: Investigate how to disable zero division error.

    # NOTE: See error note in seqeval_error.txt, which only comes up with subset of size 024, not 512 when testing.
    # This error is similar to this one from Stack: https://stackoverflow.com/questions/69596496/with-cpupytorch-indexerror-index-out-of-range-in-self-with-cudaassertion

    print(model_name + " Results: ")
    print("precision: ", test_results["overall_precision"], "\nrecall: ", test_results["overall_recall"],
          "\nf1: ", test_results["overall_f1"], "\naccuracy: ", test_results["overall_accuracy"], 
          "\ntotal_time_in_seconds: ", test_results["total_time_in_seconds"], "\nsamples_per_second: ", test_results["samples_per_second"],
          "\nlatency_in_seconds: ", test_results["latency_in_seconds"])