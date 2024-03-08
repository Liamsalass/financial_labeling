import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

def return_dataset_df(dataset, col, is_full):
    """
    Given a PyTorch dataset, return a Pandas df. Handles individual splits and full concatenated FiNER dataset.
    """
    assert col.lower() in ["tokens", "ner_tags"]
    if not is_full:
        dataset_df = dataset.data[col].to_pandas()
    else:
        # When using Torch ConcatDataset
        dataset_df_1 = dataset.datasets[0].data[col].to_pandas()
        dataset_df_2 = dataset.datasets[1].data[col].to_pandas()
        dataset_df_3 = dataset.datasets[2].data[col].to_pandas()
        dataset_df = pd.concat([dataset_df_1, dataset_df_2, dataset_df_3])

    return dataset_df

def length_histograms(dataset, dataset_name, characters, is_full):
    """
    Plots histogram of character or sentence count.
    """
    
    dataset_df = return_dataset_df(dataset, col="tokens", is_full=is_full)
    
    if characters:  # Total Characters
        dataset_df.str.len().hist()
        plt.title(dataset_name + " FiNER Dataset Character Count Histogram")
        plt.xlabel("Total Characters")
    else:  # Total Sentences
        dataset_df.apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
        plt.title(dataset_name + " FiNER Dataset Sentence Count Histogram")
        plt.xlabel("Total Sentences")
    
    
    plt.ylabel("Frequency")
    plt.show()
    # TODO: Plots without outliers?


def label_counts(dataset, dataset_name, label2id_dict, top_n, is_full):
    dataset_df = return_dataset_df(dataset, col="ner_tags", is_full=is_full)

    # Count of each ner tag in the dataset, stored in a dict
    counts = {}
    for ner_tags in dataset_df:
        for tag in ner_tags:
            if tag != 0:
                counts[tag] = counts.get(tag, 0) + 1
    # TODO: Review how to go from ner_tags to labels. Check logic

    # Merge the counts dict with the actual labels instead of tags.
    
    merged_dict = {}
    for label, id_ in label2id_dict.items():
        if id_ in counts:
            merged_dict[label] = counts[id_]
    
    sorted_labels = sorted(merged_dict.items(), key=lambda x: x[1], reverse=True)
    top_n_labels = dict(sorted_labels[:top_n])

    plt.bar(top_n_labels.keys(), top_n_labels.values())
    plt.title(f'Top {top_n} Labels from ' + dataset_name + ' FiNER Dataset')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    # TODO: Pad graph so that full labels are shown.
    

if __name__ == "__main__":
    train_dataset = datasets.load_dataset("nlpaueb/finer-139", split="train")
    val_dataset = datasets.load_dataset("nlpaueb/finer-139", split="validation")
    test_dataset = datasets.load_dataset("nlpaueb/finer-139", split="test")
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    # Sample count in each dataset
    print("Train Dataset Size: ", str(len(train_dataset)))
    print("Validation Dataset Size: ", str(len(val_dataset)))
    print("Test Dataset Size: ", str(len(test_dataset)))
    print("Full Dataset Size: ", str(len(full_dataset)))

    # length_histograms(full_dataset, dataset_name="Full", characters=True, is_full=True)
    # length_histograms(full_dataset, dataset_name="Full", characters=False, is_full=True)

    finer_tag_names = train_dataset.features["ner_tags"].feature.names
    label2id = {value: i for i, value in enumerate(finer_tag_names)}

    label_counts(full_dataset, dataset_name="Full", label2id_dict=label2id, top_n=10, is_full=True)

    # TODO: Expand on EDA?
    # Resource: https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools