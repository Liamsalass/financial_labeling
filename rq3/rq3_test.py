import datasets
from evaluate import evaluator
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
from tqdm import tqdm

def test_model(model_name):
    """
    Function which will test the pretrained SEC-BERT and the MobileBERT models on FiNER-139.
    """
    
    test_dataset = datasets.load_dataset("nlpaueb/finer-139", split="test")
    # NOTE: first 64 samples taken to see that code runs end to end on CPU. Remove this to test full thing
    test_dataset = test_dataset.select(range(64))
    # TODO: Does this need to be tokenized or does HF do it automatically? Assume it is done in the pipeline since tokenizers are attached.

    if model_name == "SEC-BERT":
        # NOTE: SEC-BERT Classifier weights and biases are not loaded. The HF pretrained model is meant for fill-masking. See if we can load these weights from somewhere?
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-base")
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")
        print("Testing SEC-BERT")
        print("SEC-BERT-BASE Parameter Count: ", model.num_parameters())
    elif model_name == "MobileBERT":
        # TODO: Need to adjust this so that it can find the best run for MobileBERT
        model = AutoModelForTokenClassification.from_pretrained("rq3_model/checkpoint-16")
        tokenizer = AutoTokenizer.from_pretrained("rq3_model/checkpoint-16")
        print("Testing MobileBERT")
        print("MobileBERT Parameter Count: ", model.num_parameters())

    
    # https://huggingface.co/docs/evaluate/en/base_evaluator
    classifier = pipeline(task="token-classification", model=model, tokenizer=tokenizer)
    task_evaluator = evaluator(task="token-classification")
    # TODO: How to batch this? How to use with GPU?
    # Possible Batching link: https://huggingface.co/docs/transformers/en/main_classes/pipelines
    test_results = task_evaluator.compute(
        model_or_pipeline=classifier,
        data=test_dataset
    )

    print(model_name + " Results: ")
    print(test_results)


print("\n")
test_model("SEC-BERT")
print("\n")
test_model("MobileBERT")
