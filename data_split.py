import random
import pandas as pd
import numpy as np

from dataset.dataset import ImageDataset

def split_train_val(train_csv, test_csv, val_csv_destination, seed = 42):
    """
    Splits a dataset into a smaller training set and a validation set.
    The validation set size = the test set size
    """

    random.seed(seed)
    np.random.seed(seed)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    test_size = len(test_df)

    # Shuffle data with a fixed random seed
    train_shuffled = train_df.sample(frac=1, random_state=seed).reset_index(drop=True) # random sample of train set (frac = 1 means all data)
    val_df = train_shuffled.iloc[:test_size]

    val_df.to_csv("./data/val_info.csv", index=False)

    print(f"Validation set size: {len(val_df)}")
    print("Validation Set Sample:")
    print(val_df.head())

if __name__ == "__main__":
    split_train_val("./data/train_info.csv", "./data/test_info.csv", seed=42)