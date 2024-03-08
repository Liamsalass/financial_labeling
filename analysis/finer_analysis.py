import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset


def length_histograms(dataset, dataset_name, characters, is_full=False):
    # Converts torch dataset to pandas, then plots histogram of character or sentence count
    if not is_full:
        dataset_df = dataset.data["tokens"].to_pandas()
    else:
        # When using Torch ConcatDataset
        dataset_df_1 = dataset.datasets[0].data["tokens"].to_pandas()
        dataset_df_2 = dataset.datasets[1].data["tokens"].to_pandas()
        dataset_df_3 = dataset.datasets[2].data["tokens"].to_pandas()
        dataset_df = pd.concat([dataset_df_1, dataset_df_2, dataset_df_3])
    
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

    length_histograms(full_dataset, dataset_name="Full", characters=True, is_full=True)
    length_histograms(full_dataset, dataset_name="Full", characters=False, is_full=True)

    # TODO: Expand on EDA?
    # Resource: https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools