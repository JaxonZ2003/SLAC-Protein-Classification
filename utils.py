# class for learning rate scheduler and early stopping, etc.
import torch
import random
import pandas as pd
import numpy as np


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluation helper function for a model to evaluate on a dataset
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(dataloader)
    test_acc = correct / total

    return test_loss, test_acc

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
def visualize_performance(train_log_path: str, out_dir: str, file_name: str) -> None:
    import json
    import matplotlib.pyplot as plt

    # Load the train log from the JSON file
    with open(train_log_path, 'r') as f:
        train_log = json.load(f)

    # variables for plotting
    epochs = train_log['epoch']
    train_loss = train_log['train_loss']
    val_loss = train_log['val_loss']
    train_acc = train_log['train_acc']
    val_acc = train_log['val_acc']

    # Check if validation data is all zeros
    if all(v == 0 for v in val_loss):
        print("Warning: Validation loss data is all zeros.")
        val_loss = None

    if all(v == 0 for v in val_acc):
        print("Warning: Validation accuracy data is all zeros.")
        val_acc = None

    # plotting
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(epochs, train_loss, label='Training Loss')
    if val_loss is not None:
        axs[0].plot(epochs, val_loss, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epochs, train_acc, label='Training Accuracy')
    if val_acc is not None:
        axs[1].plot(epochs, val_acc, label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir + '/' + file_name)
    plt.close()

    print(f"Performance plot saved to {out_dir}/{file_name}")