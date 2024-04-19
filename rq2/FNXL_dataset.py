import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from collections import Counter

class FNXLDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)

        self.sentences = self.data_frame['sentence'].values
        self.ner_tags = self.data_frame['ner_tags'].apply(lambda x: json.loads(x)).values

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        ner_tags = self.ner_tags[idx]
        sample = {'sentence': sentence, 'ner_tags': ner_tags}
        return sample

def gather_label_distribution(dataset):
    all_tags = [tag for sublist in dataset.ner_tags for tag in sublist]
    tag_counts = Counter(all_tags)
    return tag_counts


if __name__ == '__main__':
    import os 
    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument('--root',
                        help='root directory of the project',
                        type=str,
                        default='D:/FNXL/')
    
    args = parser.parse_args()

    
    train_dataset = FNXLDataset(os.path.join(args.root, 'train_sample.csv'))
    test_dataset = FNXLDataset(os.path.join(args.root, 'test_sample.csv'))

    print('Train dataset length:', len(train_dataset))
    print('Test dataset length:', len(test_dataset))

    print('Train dataset label distribution:', gather_label_distribution(train_dataset))
    print('Test dataset label distribution:', gather_label_distribution(test_dataset))
