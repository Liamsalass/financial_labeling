import datasets
from evaluate import evaluator
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline, TokenClassificationPipeline


def test_model(model_name):
    """
    Function which will test the pretrained SEC-BERT and the MobileBERT models on FiNER-139.
    """
    
    test_dataset = datasets.load_dataset("nlpaueb/finer-139", split="test")
    # NOTE: first 64 samples taken to see that code runs end to end on CPU. Remove this to test full thing
    test_dataset = test_dataset.select(range(64))
    # TODO: Does this need to be tokenized or does HF do it automatically?
    
    # Getting array of tags/labels
    finer_tag_names = test_dataset.features["ner_tags"].feature.names
    # id2label and label2id dictionaries for loading the model.
    id2label = {i: element for i, element in enumerate(finer_tag_names)}
    label2id = {value: i for i, value in enumerate(finer_tag_names)}

    if model_name == "SEC-BERT":
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-base", num_labels=279, id2label=id2label, label2id=label2id)
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")
        print("Testing SEC-BERT")
        print("SEC-BERT-BASE Parameter Count: ", model.num_parameters())
    elif model_name == "MobileBERT":
        # TODO: Need to adjust this so that it can find the best run for MobileBERT
        model = AutoModelForTokenClassification.from_pretrained("rq3_model/checkpoint-8")
        tokenizer = AutoTokenizer.from_pretrained("rq3_model/checkpoint-8")
        print("Testing MobileBERT")
        print("MobileBERT Parameter Count: ", model.num_parameters())

    
    # https://huggingface.co/docs/evaluate/en/base_evaluator
    classifier = pipeline(task="token-classification", model=model, tokenizer=tokenizer)
    task_evaluator = evaluator(task="token-classification")
    # TODO: How to batch this?
    # Batching link: https://huggingface.co/docs/datasets/en/how_to_metrics
    test_results = task_evaluator.compute(
        model_or_pipeline=classifier,
        data=test_dataset
    )

    print(model_name + " Results: ")
    print(test_results)
    # TODO: Save these results


test_model("SEC-BERT")
test_model("MobileBERT")