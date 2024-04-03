import datasets
from evaluate import evaluator
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
from rq3_utils import sec_bert_num_preprocess, sec_bert_shape_preprocess

def test_model(model_name):
    """
    Function which will test the pretrained SEC-BERT and the MobileBERT models on FiNER-139.
    """
    
    test_dataset = datasets.load_dataset("nlpaueb/finer-139", split="test")
    # NOTE: first few samples taken to see that code runs end to end on CPU. Remove this to test full thing
    test_dataset = test_dataset.select(range(128))
    # TODO: Does this need to be tokenized or does HF do it automatically? Assume it is done in the pipeline since tokenizers are attached.
    if model_name == "SEC-BERT-BASE":
        # NOTE: SEC-BERT Classifier weights and biases are not loaded. The HF pretrained model is meant for fill-masking. See if we can load these weights from somewhere?
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-base")
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")
        print("Testing SEC-BERT-BASE")
        print("SEC-BERT-BASE Parameter Count: ", model.num_parameters())
    elif model_name == "SEC-BERT-NUM":
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-num")
        test_dataset = test_dataset.map(sec_bert_num_preprocess, batched=True)  # Apply SEC-BERT-NUM preprocessing
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-num")
        print("Testing SEC-BERT-NUM")
        print("SEC-BERT-NUM Parameter Count: ", model.num_parameters())
    elif model_name == "SEC-BERT-SHAPE":
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-shape")
        test_dataset = test_dataset.map(sec_bert_shape_preprocess, batched=True)  # Apply SEC-BERT-SHAPE preprocessing
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-shape")
        print("Testing SEC-BERT-SHAPE")
        print("SEC-BERT-SHAPE Parameter Count: ", model.num_parameters())
    elif model_name == "MobileBERT":
        # TODO: Need to adjust this so that it can find the best run for MobileBERT
        mobilebert_best_run = "rq3_model/checkpoint-32"
        model = AutoModelForTokenClassification.from_pretrained(mobilebert_best_run)
        tokenizer = AutoTokenizer.from_pretrained(mobilebert_best_run)
        print("Testing MobileBERT")
        print("MobileBERT Parameter Count: ", model.num_parameters())

    # TODO: Batch this?
    # Possible Batching link: https://huggingface.co/docs/transformers/en/main_classes/pipelines
    # https://stackoverflow.com/questions/75932605/getting-the-input-text-from-transformers-pipeline
    classifier_pipeline = pipeline(task="token-classification", model=model, tokenizer=tokenizer)
    task_evaluator = evaluator("token-classification")
    test_results = task_evaluator.compute(model_or_pipeline=classifier_pipeline, data=test_dataset, metric="seqeval")
    # TODO: Look into the zero_division parameter for recall and F-score

    print(model_name + " Results: ")
    # TODO: Save test results somewhere
    print("precision: ", test_results["overall_precision"], "\nrecall: ", test_results["overall_recall"], 
          "\nf1: ", test_results["overall_f1"], "\naccuracy: ", test_results["overall_accuracy"], 
          "\ntotal_time_in_seconds: ", test_results["total_time_in_seconds"], "\nsamples_per_second: ", test_results["samples_per_second"], "\nlatency_in_seconds: ", test_results["latency_in_seconds"])

test_model("SEC-BERT-BASE")
print("\n")
test_model("SEC-BERT-NUM")
print("\n")
test_model("SEC-BERT-SHAPE")
print("\n")
test_model("MobileBERT")
