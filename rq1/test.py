import datasets
import argparse
import os
import torch
import wandb
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
from peft import PeftModel, PeftConfig
from utils.tokenize_and_align import tokenize_and_align_labels_mobilebert, tokenize_and_align_labels_sec_bert_base, tokenize_and_align_labels_sec_bert_num, tokenize_and_align_labels_sec_bert_shape
from utils.metrics import compute_metrics

if __name__ == "__main__":
    wandb.init(mode="disabled")  # Disable wandb for this file.

    print("CUDA available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA current device: ", torch.cuda.current_device())  # CPU is -1. Else GPU
    else:
        print("CUDA unavailable, using CPU")

    test_dataset = datasets.load_dataset("nlpaueb/finer-139", split="test")

    # Parsing command line args
    parser = argparse.ArgumentParser(description='CMPE 351 RQ3 Testing code')
    parser.add_argument('-model_name', type=str, default='MobileBERT', help='Selected model to test. Enter one of "MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"')
    parser.add_argument('-subset', type=int, default=-1, help='Specify to use a subset of the test set. If left empty, use the entire test set.')
    parser.add_argument('-checkpoint_path', type=str, default="rq3_model/checkpoint-32", help='Specify the relative path to the Hugging Face model checkpoint to evaluate.')
    parser.add_argument('-save_results', type=bool, default=True, help='Specify whether or not to save the test metrics to a file.')
    parser.add_argument('-batch_size', type=int, default=16, help='Batch size per device')
    parser.add_argument('-peft', type=int, default=1, help='Specify whether or not the checkpoint model used PEFT during training [0/1].')
    arguments = parser.parse_args()
    
    model_name = arguments.model_name
    subset_size = arguments.subset
    checkpoint_path = arguments.checkpoint_path
    save_results = arguments.save_results
    batch_size_per_device = arguments.batch_size
    using_peft = arguments.peft

    # Verifying command line args
    assert model_name in ["MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"]

    if subset_size != -1:
        assert 0 < subset_size < len(test_dataset)
        test_dataset = test_dataset.select(range(subset_size))  # Selects the specified # of samples from the subset argument.
        print("Testing " + model_name + " on subset of FiNER-139 test set with " + str(subset_size) + " samples.")
    else:
        print("Testing " + model_name + " on full FiNER-139 test set with " + str(len(test_dataset)) + " samples.")
    
    # NOTE: Check Below doesn't check for the contents of the directory. Could possibly verify this to ensure that the weights are in the folder.
    full_checkpoint_path = os.getcwd() + "/" + checkpoint_path
    assert os.path.isdir(full_checkpoint_path) is True  # Verify checkpoint dir exists
    print("Testing " + model_name + " checkpoint stored in the " + checkpoint_path + " folder")

    assert 1 <= batch_size_per_device <= len(test_dataset)

    assert using_peft in [0, 1]

    if using_peft == 1:
        finer_tag_names = test_dataset.features["ner_tags"].feature.names

        # id2label and label2id dictionaries for loading the model.
        id2label = {i: element for i, element in enumerate(finer_tag_names)}
        label2id = {value: i for i, value in enumerate(finer_tag_names)}

        config = PeftConfig.from_pretrained(checkpoint_path)
        inference_model = AutoModelForTokenClassification.from_pretrained(
        config.base_model_name_or_path, num_labels=279, id2label=id2label, label2id=label2id)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(inference_model, checkpoint_path)
    else:  # No PEFT
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Tokenize data according to different model variants
    if model_name == "MobileBERT":
        tokenized_test = test_dataset.map(tokenize_and_align_labels_mobilebert, batched=True)
    elif model_name == "SEC-BERT-BASE":
        tokenized_test = test_dataset.map(tokenize_and_align_labels_sec_bert_base, batched=True)
    elif model_name == "SEC-BERT-NUM":
        tokenized_test = test_dataset.map(tokenize_and_align_labels_sec_bert_num, batched=True)
    elif model_name == "SEC-BERT-SHAPE":
        tokenized_test = test_dataset.map(tokenize_and_align_labels_sec_bert_shape, batched=True)
    
    print(model_name + " Parameter Count: ", model.num_parameters())

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        per_device_eval_batch_size=batch_size_per_device
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Model Trainer object, used for evaluation. Used this object since it is modular (for testing each model type),
    #  and was having issues using HF pipelines and tokenized data.
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    test_results = trainer.evaluate(eval_dataset=tokenized_test)

    # TODO: Latency metric?, possibly add batch size in reporting?
    # Print to console
    print(model_name + " model from checkpoint: " + checkpoint_path + "\nResults: \nmicro/overall precision: ",
            test_results["eval_micro/overall precision"], "\nmicro/overall recall: ", test_results["eval_micro/overall recall"],
            "\nmicro/overall f1: ", test_results["eval_micro/overall f1"], "\noverall accuracy: ", test_results["eval_overall accuracy"], 
            "\nmacro precision: ", test_results["eval_macro precision"], "\nmacro recall: ", test_results["eval_macro recall"],
            "\nmacro f1: ", test_results["eval_macro f1"], "\ntotal_time_in_seconds: ", test_results["eval_runtime"],
            "\nsamples_per_second: ", test_results["eval_samples_per_second"])
    

    if save_results:
        # Make file name for results file
        if subset_size == -1:
            results_full_path = os.getcwd() + "/results/" + model_name + "-test-results-full.txt"
        else:
            results_full_path = os.getcwd() + "/results/" + model_name + "-test-results-subset-" + str(subset_size) + ".txt"
        
        # Attempt to write results to txt file.
        try:
            with open(results_full_path, "w") as results_file:
                print(model_name + " model from checkpoint: " + checkpoint_path + "\nResults: \nmicro/overall precision: ",
                        test_results["eval_micro/overall precision"], "\nmicro/overall recall: ", test_results["eval_micro/overall recall"],
                        "\nmicro/overall f1: ", test_results["eval_micro/overall f1"], "\noverall accuracy: ", test_results["eval_overall accuracy"], 
                        "\nmacro precision: ", test_results["eval_macro precision"], "\nmacro recall: ", test_results["eval_macro recall"],
                        "\nmacro f1: ", test_results["eval_macro f1"], "\ntotal_time_in_seconds: ", test_results["eval_runtime"],
                        "\nsamples_per_second: ", test_results["eval_samples_per_second"], file=results_file)
        except:
            print("Failed to save test metrics to the file: " + results_full_path)