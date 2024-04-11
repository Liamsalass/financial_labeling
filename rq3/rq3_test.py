import datasets
import argparse
import os
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
from peft import PeftModel, PeftConfig
from rq3_utils import tokenize_and_align_labels_mobilebert, sec_bert_num_preprocess, sec_bert_shape_preprocess, compute_metrics, return_mobilebert_peft_config

if __name__ == "__main__":
    print("CUDA available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA current device: ", torch.cuda.current_device())  # CPU is -1. Else GPU
    else:
        print("CUDA unavailable, using CPU")

    # Loading the test dataset. NOTE: In future, can possibly specify path to this dataset with command line args. Not needed right now.
    test_dataset = datasets.load_dataset("nlpaueb/finer-139", split="test")
    # TODO: Do we need to map the dataset like we did in rq3_train with tokenize and align labels?

    # Parsing command line args
    parser = argparse.ArgumentParser(description='CMPE 351 RQ3 Testing code')
    parser.add_argument('-model_name', type=str, default='MobileBERT', help='Selected model to test. Enter one of "MobileBERT", "SEC-BERT-BASE", "SEC-BERT-NUM", "SEC-BERT-SHAPE"')
    parser.add_argument('-subset', type=int, default=-1, help='Specify to use a subset of the test set. If left empty, use the entire test set.')
    parser.add_argument('-checkpoint_path', type=str, default="rq3_model/checkpoint-32", help='Specify the relative path to the Hugging Face model checkpoint to evaluate.')
    parser.add_argument('-save_results', type=bool, default=True, help='Specify whether or not to save the test metrics to a file.')
    parser.add_argument('-batch_size', type=int, default=16, help='Batch size per device')
    parser.add_argument('-peft', type=bool, default=True, help='Specify whether or not the model from the checkpoint was using PEFT.')
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
    
    if model_name == "MobileBERT":
        # NOTE: Check Below doesn't check for the contents of the directory. Could possibly verify this to ensure that the weights are in the folder.
        # NOTE: Checkpoint paths are only being used for MobileBERT. May need to use them for the SEC-BERT models since these don't come with classifier weights.
        full_checkpoint_path = os.getcwd() + "/" + checkpoint_path
        assert os.path.isdir(full_checkpoint_path) is True  # Verify checkpoint dir exists
        print("Testing " + model_name + " checkpoint stored in the " + checkpoint_path + " folder")

    assert 1 <= batch_size_per_device <= len(test_dataset)

    if using_peft is True:
        # NOTE: No PEFT for SEC-BERT models for now. Revisit- may need to train with PEFT to train SEC-BERT family
        assert model_name == "MobileBERT"

    if model_name == "MobileBERT":
        if using_peft is True:
            # Getting array of tags/labels
            finer_tag_names = test_dataset.features["ner_tags"].feature.names

            # id2label and label2id dictionaries for loading the model.
            id2label = {i: element for i, element in enumerate(finer_tag_names)}
            label2id = {value: i for i, value in enumerate(finer_tag_names)}
            # TODO: Verify if these dicts are the same when finer_tag_names uses train, rather than test?

            config = PeftConfig.from_pretrained(checkpoint_path)
            inference_model = AutoModelForTokenClassification.from_pretrained(
            config.base_model_name_or_path, num_labels=279, id2label=id2label, label2id=label2id)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(inference_model, checkpoint_path)
        else:
            model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        tokenized_test = test_dataset.map(tokenize_and_align_labels_mobilebert, batched=True)
    elif model_name == "SEC-BERT-BASE":
        # NOTE: SEC-BERT Classifier weights and biases are not loaded. The HF pretrained model is meant for fill-masking. See if we can load these weights from somewhere?
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-base")
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")
    elif model_name == "SEC-BERT-NUM":
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-num")
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-num")
        tokenized_test = test_dataset.map(sec_bert_num_preprocess, batched=True)  # Apply SEC-BERT-NUM preprocessing
    elif model_name == "SEC-BERT-SHAPE":
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-shape")
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-shape")
        tokenized_test = test_dataset.map(sec_bert_shape_preprocess, batched=True)  # Apply SEC-BERT-SHAPE preprocessing
    
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

    # TODO: Latency metric, possibly add batch size in reporting?
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