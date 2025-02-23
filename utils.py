# class for learning rate scheduler and early stopping, etc.
import torch
import random
import pandas as pd
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop_training = False

    def __call__(self, epoch_loss):
        if self.best_loss is None:
            self.best_loss = epoch_loss
        elif epoch_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True


def split_train_val(train_csv, test_csv, val_csv_destination: str, seed: int = 42):
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

    val_df.to_csv(val_csv_destination, index=False)

    print(f"Validation set size: {len(val_df)}")
    print("Validation Set Sample:")
    print(val_df.head())

# create a function for visualizing the performance of the model via train log from the json file
def visualize_performance(train_log_path, out_dir: str):
    import json
    import matplotlib.pyplot as plt

    # Load the train log from the JSON file
    with open(train_log_path, 'r') as f:
        train_log = json.load(f)

    # variables for plotting
    epochs = train_log['epochs']
    train_loss = train_log['train_loss_per_epoch']
    val_loss = train_log['val_loss_per_epoch']
    train_acc = train_log['train_acc_per_epoch']
    val_acc = train_log['val_acc_per_epoch']

    # plotting
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(epochs, train_loss, label='Training Loss')
    axs[0].plot(epochs, val_loss, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epochs, train_acc, label='Training Accuracy')
    axs[1].plot(epochs, val_acc, label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir + '/performance_plot.png')
    plt.close()

    print(f"Performance plot saved to {out_dir}/performance_plot.png")
    