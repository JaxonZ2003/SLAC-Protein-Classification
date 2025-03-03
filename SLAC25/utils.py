# class for learning rate scheduler and early stopping, etc.
import torch
import random
import pandas as pd
import numpy as np
import os


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

if __name__ == "__main__":
    package_root = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(package_root, "..", "data", "train_info.csv")
    train_path = os.path.abspath(train_path)
    test_path = os.path.join(package_root, "..", "data", "test_info.csv")
    test_path = os.path.abspath(test_path)
    des_path = os.path.join(package_root, "..", "data", "val_info.csv")
    des_path = os.path.abspath(des_path)
    split_train_val(train_path, test_path, des_path, seed=42)
    # print(train_path)
    # print(test_path)