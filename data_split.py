import random
import pandas as pd

from dataset import ImageDataset

def split_train_val(train_csv, test_csv):
    """
    Splits a dataset into a smaller training set and a validation set.
    The validation set size = the test set size
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    test_size = len(test_df)

    train_shuffled = train_df.sample(frac = 1).reset_index(drop=True)
    val_df = train_shuffled.iloc[:test_size]
    new_train_df = train_shuffled.iloc[test_size:]

    new_train_df.to_csv("./data/new_train_info.csv", index=False)
    val_df.to_csv("./data/val_info.csv", index=False)


    print(f"New training set size:{len(new_train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print("Validation Set Sample:")
    print(val_df.head())
    print("New Training Set Sample:")
    print(new_train_df.head())

if __name__ == "__main__":
    split_train_val("./data/train_info.csv", "./data/test_info.csv")
