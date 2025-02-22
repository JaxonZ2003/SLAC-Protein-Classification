import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchmetrics import AUROC
from dataset import ImageDataset
from dataloader import ImageDataLoader
from data_split import split_train_val
# model imports
from Models import CNN, ResNet

def fit(model, train_loader, val_loader, num_epochs, optimizer, criterion, device, save_every=5, lr_scheduler=None, outdir='./models'):
    print(f'{"#"*10} Starting training on {device} {"#"*10}')
    # key can be the epoch number
    os.makedirs(outdir, exist_ok=True)
    train_log = {
        'train_loss_per_epoch': [],
        'train_acc_per_epoch': [],
        'val_loss_per_epoch': [],
        'val_acc_per_epoch': [],
        'learning_rates': [],
        'epochs': [],  # Add explicit epoch tracking
        'model_checkpoints': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            
            # Update running stats
            running_loss += loss.item()
            _, predicted = outputs.max(1) # gets the class with the highest probability
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() # if the predicted label equals the actual label, add 1 to the correct
            
        
        # Calculate epoch stats
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_log['epochs'].append(epoch + 1)
        train_log['train_loss_per_epoch'].append(epoch_loss)
        train_log['train_acc_per_epoch'].append(epoch_acc)

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        train_log['val_loss_per_epoch'].append(val_loss)
        train_log['val_acc_per_epoch'].append(val_acc)

        train_log['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # save model _save_every_ epochs
        if (epoch + 1) % save_every == 0:
            model_name = os.path.join( outdir, f"model_ep{epoch+1}.net")
            save_file = os.path.abspath(model_name) 
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_loss,
                    'train_acc': epoch_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, save_file)
                train_log['model_checkpoints'].append(save_file)
                print(f"Model saved to {save_file}")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        # update learning scheduler
        if lr_scheduler:
            lr_scheduler.step(val_loss)

        # save the train log to a file
        log_file = f"{outdir}/train_log.json"
        with open(log_file, "w") as f:
            json.dump(train_log, f, indent=4)

        print(f'Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_log

def evaluate_model(model, dataloader, criterion, device):
    """Helper function to evaluate the model on a dataset"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # no need to compute gradients
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) # compute the loss
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), correct / total

def test(model, dataloader, criterion, device, outdir='./models'):
    model.eval()
    # eval the model on test set
    test_loss, test_acc = evaluate_model(model, dataloader, criterion, device)
    
    # save as a dictionary
    test_log = {
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }
    
    # load existing log file...if it exists
    log_file = f"{outdir}/train_log.json"
    try:
        with open(log_file, "r") as f:
            full_log = json.load(f)
    except FileNotFoundError:
        full_log = {}
    
    # append the test log to the full log
    full_log['test_log'] = test_log

    # save the full log to a file
    with open(log_file, "w") as f:
        json.dump(full_log, f, indent=4)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    return test_log


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("nepoch", type=int)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--lr", type=float, default=0.001)
    args = ap.parse_args()

    if args.outdir is None:
        
        try:
            slurm_jid = os.environ['SLURM_JOB_ID']
            slurm_jname = os.environ['SLURM_JOB_NAME']
            username = os.environ['USER']
            args.outdir = f"/scratch/slac/models/{username}.{slurm_jname}.{slurm_jid}"
        except KeyError:
            args.outdir = "./models"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # simple model
    model = CNN(num_classes=4, keep_prob=0.75)
    model.to(device)

    # split train set into train and val
    split_train_val('./data/train_info.csv', './data/test_info.csv', './data/val_info.csv', seed=42)

    # dataset paths
    csv_train_file = './data/train_info.csv'
    csv_test_file = './data/test_info.csv'
    csv_val_file = './data/val_info.csv'

    # load the datasets
    train_dataset = ImageDataset(csv_train_file)
    test_dataset = ImageDataset(csv_test_file)
    val_dataset = ImageDataset(csv_val_file)

    train_loader = ImageDataLoader(train_dataset).get_loader()
    test_loader = ImageDataLoader(test_dataset).get_loader()
    val_loader = ImageDataLoader(val_dataset).get_loader()

    criterion = nn.CrossEntropyLoss() # internally computes the softmax so no need for it. 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
    
    # train the model
    train_log = fit(model, train_loader, val_loader, 
                    num_epochs=args.nepoch, optimizer=optimizer, criterion=criterion, device=device,
                    lr_scheduler=lr_sched, outdir=args.outdir)
    test_log = test(model, test_loader, criterion, device)