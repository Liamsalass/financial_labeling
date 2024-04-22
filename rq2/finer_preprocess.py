import os
import numpy as np
import datasets
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
from logzero import logger

nltk.download('punkt')

def tokenize(sentence: str):
    # Tokenize the sentence
    return ' '.join(token.lower() for token in word_tokenize(sentence) if token.isalnum())

def main():
    logger.info('Loading FINER-139 dataset')
    dataset = datasets.load_dataset("nlpaueb/finer-139", split='train')

    output_text_path = 'finer_texts.txt'
    output_label_path = 'finer_labels.txt'

    with open(output_text_path, 'w', encoding='utf-8') as text_file, \
         open(output_label_path, 'w', encoding='utf-8') as label_file:
        for sample in tqdm(dataset, desc='Processing', leave=False):
            # Concatenate tokens into a single string and tokenize
            text = tokenize(' '.join(sample['tokens']))
            # Convert tag indices to space-separated string
            labels = ' '.join(map(str, sample['ner_tags']))
            
            print(text, file=text_file)
            print(labels, file=label_file)
    
    logger.info('Finished preparing FINER-139 dataset for AttentionXML')

if __name__ == '__main__':
    main()
