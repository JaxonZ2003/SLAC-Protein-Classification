import json
import os
import torch
import sys
import math
import time
import pytz
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchmetrics import AUROC
from datetime import datetime
from SLAC25.dataset import ImageDataset
from SLAC25.dataloader import DataLoaderFactory
from SLAC25.utils import split_train_val, evaluate_model
from SLAC25.models import BaselineCNN
# model imports

class Wrapper:
    def __init__(self):
        self.model = BaselineCNN(num_classes=4, keep_prob=0.75)
        self.num_epochs = 10
        self.criterion = nn.CrossEntropyLoss() # internally computes the softmax so no need for it. 
        self.optimizer = optim.Adam(self.model.parameters())
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6)
        self._prepareDataLoader()
        self.outdir = './models'
        self.verbose = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepareDataLoader(self, batch_size=32, testmode=False):
        fileDir = os.path.dirname(os.path.abspath(__file__))
        if fileDir.endswith("SLAC25"):
            trainDataPath = os.path.join(fileDir, "..", "data", "train_info.csv")
            testDataPath = os.path.join(fileDir, "..", "data", "test_info.csv")
            valDataPath = os.path.join(fileDir, "..", "data", "val_info.csv")
            trainDataPath = os.path.abspath(trainDataPath)
            testDataPath = os.path.abspath(testDataPath)
            valDataPath = os.path.abspath(valDataPath)
        
        elif fileDir.endswith("capstone-SLAC"):
            trainDataPath = os.path.join(fileDir, "data", "train_info.csv")
            testDataPath = os.path.join(fileDir, "data", "test_info.csv")
            valDataPath = os.path.join(fileDir, "data", "val_info.csv")
            trainDataPath = os.path.abspath(trainDataPath)
            testDataPath = os.path.abspath(testDataPath)
            valDataPath = os.path.abspath(valDataPath)
        
        trainDataset = ImageDataset(trainDataPath)
        testDataset = ImageDataset(testDataPath)
        valDataset = ImageDataset(valDataPath)

        if testmode: # take only first 50 data
            trainSubDataset = Subset(trainDataset, list(range(50)))
            testSubDataset = Subset(testDataset, list(range(50)))
            valSubDataset = Subset(valDataset, list(range(50)))
            train_factory = DataLoaderFactory(trainSubDataset, batch_size=5)
            test_factory = DataLoaderFactory(testSubDataset, batch_size=5)
            val_factory = DataLoaderFactory(valSubDataset, batch_size=5)
        
        else:
            train_factory = DataLoaderFactory(trainDataset, batch_size)
            test_factory = DataLoaderFactory(testDataPath, batch_size)
            val_factory = DataLoaderFactory(valDataset, batch_size)

        train_factory.setSequentialSampler()
        test_factory.setSequentialSampler()
        val_factory.setSequentialSampler()

        self.train_loader = train_factory.outputDataLoader()
        self.test_loader = test_factory.outputDataLoader()
        self.val_loader = val_factory.outputDataLoader()


class BaselineCNN_Wrapper(Wrapper):
    def __init__(self, num_classes=4, keep_prob=0.75, num_epochs=10, outdir='./models', verbose=False):
        super().__init__()
        self.model = BaselineCNN(num_classes, keep_prob)
        self.model.to(self.device)
        self.num_epochs = num_epochs
        self.outdir = outdir
        self.verbose = verbose
        
      
    def train(self):
        print('{} Starting training on {} {}'.format('-'*10, self.device, '-'*10))
        print(f"Total epochs: {self.num_epochs}")

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
        
        # training loop
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad() # zero the gradients
                outputs = self.model(images) # forward pass
                loss = self.criterion(outputs, labels) # compute the loss
                loss.backward() # backpropagation
                self.optimizer.step() # update the weights

                if self.verbose:
                    print(f'Epoch: {epoch + 1}/{self.num_epochs} | Batch: {batch_idx + 1} | Loss: {loss:.3f}')
                
                if loss is None or math.isnan(loss) or math.isinf(loss):
                    print(f"Error: Loss became undefined or infinite at Epoch: {epoch + 1} | Batch: {batch_idx + 1}.")
                    print(f"Stopping training.")
                    sys.exit(1)
                
                # Update running stats
                running_loss += loss.item() # extract the tensor and return a float
                _, predicted = outputs.max(1) # gets the class with the highest probability
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item() # if the predicted label equals the actual label, add 1 to the correct
                
            
            if self.verbose:
                print(f"{'-'*10} Training Epoch {epoch + 1} finished {'-'*10}")
            # Calculate epoch stats
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct / total
            train_log['epoch'].append(epoch + 1)
            train_log['train_loss'].append(epoch_loss)
            train_log['train_acc'].append(epoch_acc)

            if self.verbose:
                print(f"Training Epoch {epoch + 1} Average Loss: {epoch_loss} | Accuracy: {epoch_acc}")
                print(f"{'-'*10} Start Validation {epoch + 1} {'-'*10}")

            # After training phase: Validation phase
            self.model.eval()
            min_val_loss = float('inf')
            val_running_loss = 0.0  # Separate variable for validation
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_running_loss / len(self.val_loader)
            val_acc = val_correct / val_total

            # update the minimum validation loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss

            # update the validation loss and accuracy
            train_log['val_loss'].append(val_loss)
            train_log['val_acc'].append(val_acc)

            if self.verbose:
                print(f"Validation {epoch + 1} Average Loss: {val_loss} | Accuracy: {val_acc}")
                print(f"{'-'*20}")

            # update the learning rate
            train_log['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            end_time = time.time()
            time_per_epoch = end_time - start_time
            train_log['time_per_epoch'].append(time_per_epoch)

            # save model if the loss is the lowest so far
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                timeNow = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d%H%M%S")
                model_name = os.path.join(self.outdir, f"BaselineCNN_{timeNow}_ep{epoch+1}.net")
                save_file = os.path.abspath(model_name) 
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': epoch_loss,
                        'train_acc': epoch_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }, save_file)
                    train_log['model_checkpoints'].append(save_file)
                    print("New best model saved to {}".format(save_file))
                except Exception as e:
                    print("Error saving model: {}".format(e))
            
            # update the learning rate scheduler
            self.lr_scheduler.step(val_loss)

            # save the train log to a file
            log_file = f"{self.outdir}/train_log.json"
            with open(log_file, "w") as f:
                json.dump(train_log, f, indent=4)

            print('Epoch {} - Train Loss: {:.4f}, Train Acc: {:.4f} - Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch+1, epoch_loss, epoch_acc, val_loss, val_acc))
        return train_log
    
    def test(self):
        test_loss, test_acc = evaluate_model(self.model, self.test_loader, self.criterion, self.device)
        
        # save as a dictionary
        test_log = {
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }

        # load existing log file...if it exists
        log_file = f"{self.outdir}/train_log.json"
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
          
          
        


def fit(model, train_loader, val_loader, num_epochs, optimizer, criterion, device, lr_scheduler, save_every=5, outdir='./models', verbose=False):
    """
    Training loop for the model
    """
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

    print('{} Starting training on {} {}'.format('-'*10, device, '-'*10))
    print(f"Total epochs: {num_epochs}")
    # key can be the epoch number
    os.makedirs(outdir, exist_ok=True)

    # training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # zero the gradients
            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) # compute the loss
            loss.backward() # backpropagation
            optimizer.step() # update the weights

            if verbose:
                print(f'Epoch: {epoch + 1}/{num_epochs} | Batch: {batch_idx + 1} | Loss: {loss:.3f}')
            
            if loss is None or math.isnan(loss) or math.isinf(loss):
                print(f"Error: Loss became undefined or infinite at Epoch: {epoch + 1} | Batch: {batch_idx + 1}.")
                print(f"Stopping training.")
                sys.exit(1)
            
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

        # After training phase: Validation phase
        model.eval()
        min_val_loss = float('inf')
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

        # update the validation loss and accuracy
        train_log['val_loss'].append(val_loss)
        train_log['val_acc'].append(val_acc)

        # update the learning rate
        train_log['learning_rates'].append(optimizer.param_groups[0]['lr'])

        end_time = time.time()
        time_per_epoch = end_time - start_time
        train_log['time_per_epoch'].append(time_per_epoch)

        # save model if the loss is the lowest so far
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model_name = os.path.join(outdir, f"model_ep{epoch+1}.net")
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
        
        # update the learning rate scheduler
        lr_scheduler.step(val_loss)

        # save the train log to a file
        log_file = f"{outdir}/train_log.json"
        with open(log_file, "w") as f:
            json.dump(train_log, f, indent=4)

        print('Epoch {} - Train Loss: {:.4f}, Train Acc: {:.4f} - Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch+1, epoch_loss, epoch_acc, val_loss, val_acc))
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

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    return test_log


if __name__ == "__main__":
    ########## parse arguments #########
    from argparse import ArgumentParser
    ap = ArgumentParser()
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
    test_dataset = ImageDataset(csv_test_file)
    val_dataset = ImageDataset(csv_val_file)

    train_factory = DataLoaderFactory(train_dataset, num_workers=4)
    train_factory.setSequentialSampler()
    test_factory = DataLoaderFactory(test_dataset, num_workers=4)
    test_factory.setSequentialSampler()
    val_factory = DataLoaderFactory(val_dataset, num_workers=4)
    val_factory.setSequentialSampler()

    train_loader = train_factory.outputDataLoader()
    test_loader = test_factory.outputDataLoader()
    val_loader = val_factory.outputDataLoader()

    criterion = nn.CrossEntropyLoss() # internally computes the softmax so no need for it. 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6)
    
    
    ########## train the model #########
    train_log = fit(model, train_loader, val_loader, num_epochs=args.nepoch, 
                    optimizer=optimizer, criterion=criterion, device=device, 
                    lr_scheduler=lr_scheduler, outdir=args.outdir)
    #test_log = test(model, test_loader, criterion, device) # will save testing for later