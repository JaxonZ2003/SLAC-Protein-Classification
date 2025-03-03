import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_main import ImageDataset
from sampler import EqualGroupSampler
from dataloader import ImageDataLoader
from data_split import split_train_val
from utils import *
import time
# model imports
from Models import CNN, ResNet

def fit(model, train_loader, num_epochs, optimizer, criterion, device, lr_scheduler=None, val_loader=None, early_stopping=None, outdir='./models'):
    """
    Training loop for the model
    """
    print('{} Starting training on {} {}'.format('-'*10, device, '-'*10))
    # key can be the epoch number
    os.makedirs(outdir, exist_ok=True)
    train_log = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'model_checkpoints': [],
        'time_per_epoch': []
    }
    # init. min. val. loss to infinity
    min_val_loss = float('inf')

    # initialize early stopping
    if early_stopping is not None:
        print("Early Stopping is enabled")
    

    # training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        print(f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # zero the gradients
            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) # compute the loss
            loss.backward() # backpropagation
            optimizer.step() # update the weights
            
            # Update running stats
            running_loss += loss.item()
            _, predicted = outputs.max(1) # gets the class with the highest probability
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() # if the predicted label equals the actual label, add 1 to the correct
            
        
        # Calculate epoch stats
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_log['epoch'].append(epoch + 1)
        train_log['train_loss'].append(epoch_loss)
        train_log['train_acc'].append(epoch_acc)

        # After training phase
        if val_loader is not None: # if there is a validation set
            model.eval()
            val_running_loss = 0.0  # Separate variable for validation
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_running_loss / len(val_loader)
            val_acc = val_correct / val_total

            # update the minimum validation loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # save model if the loss is the lowest so far
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
                    print("New best model saved to {}".format(save_file))
                except Exception as e:
                    print("Error saving model: {}".format(e))


            # update the validation loss and accuracy
            train_log['val_loss'].append(val_loss)
            train_log['val_acc'].append(val_acc)

        # if no validation set, skip the validation phase and fill in 0's
        else:
            val_loss = 0
            val_acc = 0
            train_log['val_loss'].append(val_loss)
            train_log['val_acc'].append(val_acc)


        # update the learning rate
        train_log['learning_rates'].append(optimizer.param_groups[0]['lr'])

        end_time = time.time()
        time_per_epoch = end_time - start_time
        train_log['time_per_epoch'].append(time_per_epoch)
        
        # update the learning rate scheduler
        if val_loader is not None and lr_scheduler is not None:
            print("Stepping learning rate scheduler by a factor of {}".format(lr_scheduler.factor))
            lr_scheduler.step(val_loss)

        # check if early stopping is triggered
        if val_loader is not None and early_stopping is not None:
            early_stopping(val_loss, model)
            if early_stopping.early_stop: # flag that is raised or not
                print("Early stopping")
                break # stop training

        # save the train log to a file
        log_file = f"{outdir}/train_log.json"
        with open(log_file, "w") as f:
            json.dump(train_log, f, indent=4)

        print('Epoch {} - Train Loss: {:.4f}, Train Acc: {:.4f} - Val Loss: {:.4f}, Val Acc: {:.4f}, Time Taken: {:.2f}s'.format(epoch+1, epoch_loss, 100.0*epoch_acc, val_loss, 100.0*val_acc, time_per_epoch))
    print('{} Training complete! {}'.format('-'*10, '-'*10))
    return train_log

def test(model, dataloader, criterion, device, outdir='./models'):
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

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {100.0*test_acc:.4f}')
    return test_log


if __name__ == "__main__":
    ########## parse arguments for the command line interface #########
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--num_workers', type=int, default=1)
    ap.add_argument("--nepoch", type=int, default=10)
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

    ########## split train set into train and val if validation set does not exist #########
    #split_train_val('./data/train_info.csv', './data/test_info.csv', './data/val_info.csv', seed=42)

    ########## dataset paths #########
    csv_train_file = './data/train_info.csv'
    csv_test_file = './data/test_info.csv'
    csv_val_file = './data/val_info.csv'

    # load the datasets
    train_dataset = ImageDataset(csv_train_file)
    train_sampler = EqualGroupSampler(train_dataset, 23299)
    if len(train_sampler) != 23299 * 4:
        raise ValueError("Train sampler length is not equal to 23299 * 4")
    test_dataset = ImageDataset(csv_test_file)
    val_dataset = ImageDataset(csv_val_file)

    train_loader = ImageDataLoader(train_sampler, num_workers=args.num_workers).get_loader()
    test_loader = ImageDataLoader(test_dataset, num_workers=args.num_workers).get_loader()
    val_loader = ImageDataLoader(val_dataset, num_workers=args.num_workers).get_loader()

    criterion = nn.CrossEntropyLoss() # internally computes the softmax so no need for it. 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=7, path=os.path.join(args.outdir, 'checkpoint.pt'), verbose=True)
    
    ########## train the model #########
    train_log = fit(model, train_loader, val_loader, num_epochs=args.nepoch, 
                    optimizer=optimizer, criterion=criterion, device=device, 
                    lr_scheduler=lr_scheduler, early_stopping=early_stopping, outdir=args.outdir)
    #test_log = test(model, test_loader, criterion, device) # will save testing for later