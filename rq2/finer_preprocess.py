
import os
import re
import numpy as np
import datasets
from tqdm import tqdm
from logzero import logger
from nltk.tokenize import word_tokenize
import nltk
import click

nltk.download('punkt')


def tokenize(sentence: str, sep='/SEP/'):
    # Tokenizes the sentence, uses /SEP/ to separate titles from descriptions if needed
    return [token.lower() if token != sep else token for token in word_tokenize(sentence)
            if len(re.sub(r'[^\w]', '', token)) > 0]

def build_vocab(sentences, vocab_size=500000):
    # Build vocabulary from tokenized sentences
    vocab = {}
    word_freq = {}
    for sentence in sentences:
        for word in sentence:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
    vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}
    return vocab

def convert_to_binary(texts, tags, max_len, vocab):
    # Converts texts and tags into a binary matrix format using the vocabulary and max length constraints
    # Assumes tags may need expansion to fit the format
    max_tags_len = max(len(tag) for tag in tags)  # Find the longest tag sequence
    text_data = np.zeros((len(texts), max_len), dtype=int)
    tag_data = np.zeros((len(texts), max_tags_len), dtype=int)  # Adjust tag data size to longest tag sequence

    for i, (text, tag) in enumerate(zip(texts, tags)):
        for j, word in enumerate(text[:max_len]):
            text_data[i, j] = vocab.get(word, 0)  # Use 0 for unknown words
        tag_data[i, :len(tag)] = tag  # Assign tag only up to the length of the current tag array

    return text_data, tag_data

@click.command()
@click.option('--output-dir', type=click.Path(), default='./processed', help='Output directory for processed files.')
@click.option('--split' , type=click.STRING, default='train', help='Dataset split to preprocess.')
@click.option('--max-len', type=click.INT, default=128, help='Maximum sequence length.')
@click.option('--vocab-size', type=click.INT, default=50000, help='Maximum vocabulary size.')

def main(output_dir, max_len, vocab_size, split):
    output_dir = output_dir + '_' + split
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info('Loading FINER-139 dataset')
    dataset = datasets.load_dataset("nlpaueb/finer-139", split=split)
    texts = [sample['tokens'] for sample in dataset]
    tags = [sample['ner_tags'] for sample in dataset]
    logger.info('Loaded {} samples'.format(len(texts)))

    logger.info('Tokenizing texts')
    tokenized_texts = [tokenize(" ".join(text)) for text in tqdm(texts, desc='Tokenizing')]
    
    logger.info('Building vocabulary')
    vocab = build_vocab(tokenized_texts, vocab_size)
    
    logger.info('Converting texts and tags to binary format')
    texts_binary, tags_binary = convert_to_binary(tokenized_texts, tags, max_len, vocab)
    
    text_output_path = os.path.join(output_dir, 'texts.npy')
    tags_output_path = os.path.join(output_dir, 'tags.npy')
    vocab_output_path = os.path.join(output_dir, 'vocab.npy')
    
    logger.info('Saving processed data')
    np.save(text_output_path, texts_binary)
    np.save(tags_output_path, tags_binary)
    np.save(vocab_output_path, list(vocab.keys()))  # Save vocabulary as a list of words

    logger.info('Finished preprocessing FINER-139 dataset')

if __name__ == '__main__':
    main()
