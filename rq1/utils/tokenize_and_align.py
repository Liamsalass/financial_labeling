from transformers import AutoTokenizer
from utils.rq3_utils import return_mobilebert_tokenizer

def tokenize_and_align_labels_mobilebert(examples):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ
    Function to align tokens and labels. 
    """
    # Load MobileBERT tokenizer.
    tokenizer = return_mobilebert_tokenizer()

    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding="max_length",  # Ensures all sequences are the same length
        max_length=512,        # Ensures no sequence exceeds model's max length
        is_split_into_words=True)

    label_all_tokens = True # Bool which is enabled to label all tokens. Otherwise, first token only.

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_and_align_labels_sec_bert_base(examples):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ
    Function to align tokens and labels for SEC-BERT-BASE
    """
    # Load SEC-BERT-BASE tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    label_all_tokens = True # Bool which is enabled to label all tokens. Otherwise, first token only.

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def tokenize_and_align_labels_sec_bert_num(examples):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ
    Function to align tokens and labels for SEC-BERT-NUM
    """
    # Load SEC-BERT-NUM tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-num")

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    label_all_tokens = True # Bool which is enabled to label all tokens. Otherwise, first token only.

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_and_align_labels_sec_bert_shape(examples):
    """
    https://huggingface.co/docs/transformers/en/tasks/token_classification
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ
    Function to align tokens and labels for SEC-BERT-SHAPE
    """
    # Load SEC-BERT-SHAPE tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-shape")

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    label_all_tokens = True # Bool which is enabled to label all tokens. Otherwise, first token only.

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs