import json
import os
import torch
import sys
import math
import time
import pytz
import tempfile
import gc
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from torchmetrics import AUROC
from datetime import datetime
from ray import tune
from ray.tune import Checkpoint

from SLAC25.dataset import ImageDataset
from SLAC25.dataloader import DataLoaderFactory
from SLAC25.utils import split_train_val, evaluate_model, EarlyStopping, find_data_path
from SLAC25.models import *

class Wrapper:
    """
    General class that applies to all models
    """
    def __init__(self, model, num_epochs, optimizer, batch_size, outdir='./models', verbose=False, testmode=False, tune=False, seed=1):
        self.model = model
        self.num_epochs = None if (num_epochs < 0) else num_epochs
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss() # internally computes the softmax so no need for it. 
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6)
        self.EarlyStopping = None # EarlyStopping(patience=7, verbose=False)
        self.tune = tune
        self.seed = seed
        
        self.verbose = verbose
        self.testmode = testmode
        self.outdir = outdir
        
        self._set_seed()
        self._prepareDataLoader(batch_size=batch_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move model to device
    
    def _set_seed(self):
        if self.tune:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False

    def _prepareDataLoader(self, batch_size=32, max_imgs=None, nwork=0):
        fileDir = os.path.dirname(os.path.abspath(__file__))
        dataPaths = find_data_path(fileDir)
        
        trainDataset = ImageDataset(dataPaths[0])
        testDataset = ImageDataset(dataPaths[1])
        valDataset = ImageDataset(dataPaths[2])

        if self.testmode is True: # take only first 50 data
            trainDataset = Subset(trainDataset, list(range(50)))
            testDataset = Subset(testDataset, list(range(10)))
            valDataset = Subset(valDataset, list(range(10)))
            batch_size = 5

        elif self.tune is True:
            g = torch.Generator()
            g.manual_seed(self.seed)

            N_TOTAL = len(trainDataset)
            N_TRAIN = 3000               
            N_VAL   = 1000                  
            assert N_TRAIN + N_VAL <= N_TOTAL, "requested split larger than datas"

            perm = torch.randperm(N_TOTAL, generator=g)
            train_idx = perm[:N_TRAIN].tolist()
            val_idx   = perm[N_TRAIN:N_TRAIN + N_VAL].tolist()
            
            # subset_indices = torch.randperm(len(trainDataset), generator=g)[:N_SAMPLES].tolist()

        else:
            if max_imgs is not None:
                ntrain = int(.9 * max_imgs)
                ntest = max_imgs - ntrain
                trainDataset = Subset(trainDataset, list(range(ntrain)))
                testDataset = Subset(testDataset, list(range(ntest)))
                valDataset = Subset(valDataset, list(range(ntest)))
        
        train_factory = DataLoaderFactory(trainDataset, batch_size, num_workers=nwork, shuffle=False)
        test_factory = DataLoaderFactory(testDataset, batch_size, num_workers=nwork, shuffle=False)
        val_factory = DataLoaderFactory(valDataset, batch_size, num_workers=nwork, shuffle=False)


        if self.tune is True:
            train_factory.setSubsetRandomSampler(train_idx, generator=g)
            val_factory.setSubsetRandomSampler(val_idx, generator=g)

            self.train_loader = train_factory.outputDataLoader()
            self.val_loader = val_factory.outputDataLoader()

        else:
            train_factory.setSequentialSampler()
            test_factory.setSequentialSampler()
            val_factory.setSequentialSampler()

            self.train_loader = train_factory.outputDataLoader()
            self.test_loader = test_factory.outputDataLoader() 
            self.val_loader = val_factory.outputDataLoader()


class Trainable(tune.Trainable):
    """
    Defining all tunable hyperparams
    """
    def setup(self, config):
        self.num_epochs = config["num_epochs"]
        self.lr = config["lr"]
        self.model = self._init_model(config["model"], config["num_classes"], config["keep_prob"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.batchsize = config["batch_size"]


        self.outdir = config["outdir"]
        self.verbose = config["verbose"]
        self.testmode = config["testmode"]
        self.seed = config["seed"]

        self.wrapper = ModelWrapper(self.model,
                                    self.num_epochs,
                                    self.optimizer,
                                    self.batchsize,
                                    self.outdir,
                                    self.verbose,
                                    self.testmode,
                                    self.seed)

        self.best_loss= float("inf")
        self.best_state = None
        
    def _init_model(self, model, num_classes, keep_prob):
        if isinstance(model, nn.Module):
            print("Using pre-instantiated model. num_classes and keep_prob are ignored.\n")
                  
        elif isinstance(model, str):
            if model == "BaselineCNN":
                model = BaselineCNN(num_classes, keep_prob)
                print(f"Instantiating new model with num_classes={model.num_classes} and keep_prob={model.keep_prob}")
        
        else:
            raise ValueError("model can either be a nn.Module or a pre-stated string.")

        assert isinstance(model, nn.Module), "Expected a PyTorch model"

        return model
    
    def step(self):
        self.wrapper._train_one_epoch()
        val_acc, val_loss = self.wrapper._val_one_epoch()
        val_acc = float(val_acc)
        val_loss = float(val_loss)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = {
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "epoch": self.iteration,
            }

        return {"val_accuracy": val_acc, "val_loss": val_loss}
        # finally:
        #     del self.model
        #     del self.optimizer
        #     gc.collect()
        #     torch.cuda.empty_cache()
        
    def save_checkpoint(self, chkpt_dir):
        """
        Ray calls this exactly once at the *end* (because of
        checkpoint_at_end=True).  Now we finally write our best model.
        """
        path = os.path.join(chkpt_dir, "best.pt")
        torch.save(self.best_state, path)
        return chkpt_dir       # Ray handles the directory
    
    def reset_config(self, new_config):
        self.config = new_config
        return True


    
class ModelWrapper(Wrapper): # inherits from Wrapper class
    def __init__(self, model, num_epochs, optimizer, batch_size, outdir='./models', verbose=False, testmode=False, tune=False, seed=1):
        super().__init__(model, num_epochs, optimizer, batch_size, outdir=outdir, verbose=verbose, testmode=testmode, tune=tune, seed=seed)
        self._verbose_printer(style="setup_header")
        self.current_epoch = 0
        self.train_log = {
          'epoch': [],
          'train_loss': [],
          'train_acc': [],
          'val_loss': [],
          'val_acc': [],
          'min_val_loss': float('inf'),
          'test_loss': [], # added for testing function during each epoch
          'test_acc': [],
          'learning_rates': [],
          'model_checkpoints': [],
          'time_per_epoch': []
        }

        self.test_log = {
            'test_loss': None,
            'test_accuracy': None
        }

    def _verbose_printer(self, style, epoch=None, batch_report=None, epoch_report=None, val_report=None, test_report=None):
        if not self.verbose:
            return

        else:
            if style == "setup_header":
                self.summary()
            
            if style == "main_header":
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                print("\n{:=^70}".format(f" Training Started on {timestamp}"))

            if style == "epoch_header":
                print("\n{:-^70}".format(f" Epoch {epoch+1}/{self.num_epochs} "))
                print("Training Phase:")
                print(f"{'Batch':>10} {'Loss':>12} {'Accuracy':>12} {'Progress':>12}")
                print("-" * 46)
            
            if style == "batch_report":
                print(f"{batch_report["batch_idx"]+1:>10d} {batch_report["loss"]:>12.4f} {batch_report["batch_acc"]:>12.4f} {batch_report["progress"]:>12}")
            
            if style == "epoch_report":
                print("\nTraining Epoch Summary:")
                print(f"Average Loss: {epoch_report["epoch_loss"]:.4f} | Accuracy: {epoch_report["epoch_acc"]:.4f}")
                print("\n{:-^70}".format(" Validation Phase "))

            if style == "val_header":
                print("\n{:-^70}".format(" Validation Phase "))

            if style == "val_report":
                print(f"Validation Results:")
                print(f"Average Loss: {val_report["val_loss"]:.4f} | Accuracy: {val_report["val_acc"]:.4f}")
                print("\n{:-^70}".format(" Testing Phase "))

            if style == "test_report":
                print(f"Testing Results:")
                print(f"Average Loss: {test_report["test_loss"]:.4f} | Accuracy: {test_report["test_acc"]:.4f}")
            
            if style == "main_footer":
                print("\n{:=^70}".format(" Training Complete "))
              
    def _testmode_operation(self):
        pass

    def summary(self):
        """
        Print a summary of the training setup configuration.

        Displays:
            - Device being used for training
            - Number of epochs and batch size
            - Output directory for saving model files
            - Whether test mode is enabled
            - Whether verbose output is enabled 
            - Whether early stopping is enabled
            - Whether learning rate scheduling is enabled
            - Optimizer and initial learning rate
        """
        print('{:#^70}'.format(''))
        print('Training Setup:\n{} Starting training on {} {}'.format('-'*10, self.device, '-'*10))
        print(f"MODEL: {self.model.__class__.__name__}")
        print(f"EPOCH COUNT: {self.num_epochs}, BATCH SIZE: {self.train_loader.batch_size}")
        print(f"SAVING MODEL TO: {self.outdir}")
        print(f"TEST MODE: {self.testmode}")
        print(f"VERBOSE: {self.verbose}")
        print(f"OPTIMIZER: {self.optimizer.__class__.__name__}, INITIAL LR: {self.optimizer.param_groups[0]['lr']}")
        # print(f"EARLY STOPPING: Enabled (patience={self.EarlyStopping.patience})")
        print(f"LEARNING RATE SCHEDULER: Enabled (patience={self.lr_scheduler.patience}, factor={self.lr_scheduler.factor})")
        if self.testmode:
            print("{} NOTE: MODEL IS RUNNING IN TEST MODE {}".format('-'*10, '-'*10))
        print('{:#^70}'.format(''))
    
    def _train_one_epoch(self):
        self.current_epoch += 1
        self._verbose_printer("epoch_header", self.current_epoch)

        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        nbatch = len(self.train_loader)
        # start_time = time.time()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad() # zero the gradients
            outputs = self.model(images) # forward pass
            loss = self.criterion(outputs, labels) # compute the loss
            loss.backward() # backpropagation
            self.optimizer.step() # update the weights
            
            if loss is None or math.isnan(loss) or math.isinf(loss):
                print(f"Error: Loss became undefined or infinite at Epoch: {self.current_epoch}/{self.num_epochs} | Batch: {batch_idx + 1}.")
                print(f"Stopping training.")
                break

            running_loss += loss.item() # extract the tensor and return a float
            _, predicted = outputs.max(1) # gets the class with the highest probability
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() # if the predicted label equals the actual label, add 1 to the correct

            if batch_idx % 100 == 0:
                progress = f"{batch_idx+1}/{nbatch}"
                batch_acc = correct / total
                batch_report = {"batch_idx": batch_idx,
                                "loss": loss.item(),
                                "batch_acc": batch_acc,
                                "progress": progress}
                
                self._verbose_printer(style="batch_report", 
                                      batch_report=batch_report)
                
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        self.train_log['epoch'].append(self.current_epoch)
        self.train_log['train_loss'].append(epoch_loss)
        self.train_log['train_acc'].append(epoch_acc)

        epoch_report = {"epoch_loss": epoch_loss,
                        "epoch_acc": epoch_acc}

        
        self._verbose_printer(style="epoch_report",
                              epoch_report=epoch_report)
        
        return epoch_acc, epoch_loss

    def _val_one_epoch(self):
        self._verbose_printer(style="val_header")
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

        if val_loss < self.train_log["min_val_loss"]:
            self.train_log["min_val_loss"] = val_loss
        
        self.train_log['val_loss'].append(val_loss)
        self.train_log['val_acc'].append(val_acc)

        val_report = {
            "val_loss": val_loss,
            "val_acc": val_acc
        }

        self._verbose_printer(style="val_report", val_report=val_report)

        if not self.testmode and val_loss < min_val_loss:
            min_val_loss = val_loss
            timeNow = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d%H%M%S")
            model_name = os.path.join(self.outdir, f"ResNet_50_Transfer_Learning_{timeNow}_ep{self.current_epoch}.net")
            save_file = os.path.abspath(model_name) 
            try:
                torch.save({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': self.train_log["epoch_loss"][-1],
                    'train_acc': self.train_log["epoch_acc"][-1],
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, save_file)
                self.train_log['model_checkpoints'].append(save_file)
                print("New best model saved to {}".format(save_file))
            except Exception as e:
                print("Error saving model: {}".format(e))
        
        self.train_log['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        ##### Learning Rate Scheduler #####
        # store the current LR
        current_lr = self.optimizer.param_groups[0]['lr']
        # check if the LR needs to be updated
        self.lr_scheduler.step(val_loss)
        # store the new LR
        new_lr = self.optimizer.param_groups[0]['lr']
        # check if the LR has been updated
        if new_lr != current_lr:
            print("Learning rate updated from {} to {}\n".format(current_lr, new_lr))

        ##### Early Stopping #####
        # check if early stopping is triggered
        if self.EarlyStopping is not None:
            self.EarlyStopping(val_loss, self.model)
            if self.EarlyStopping.early_stop: # flag that is raised or not
                print("Early stopping")
                sys.exit(1)
        
        return val_acc, val_loss

    def train(self):
        """
        Train the model.
        """
        self._verbose_printer(style="main_header")

        # training loop
        for _ in range(self.num_epochs):
            start_time = time.time()
            self._train_one_epoch()

            self._val_one_epoch()

            self.test() # make a call to the test function, returns a dictionary
            end_time = time.time()

            time_per_epoch = end_time - start_time

            # update the learning rate
            self.train_log['time_per_epoch'].append(time_per_epoch)

            # save model if the loss is the lowest so far
            
            # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            #     checkpoint = None
            #     if (epoch + 1) % 5 == 0:
            #         torch.save(
            #             self.model.state_dir(),
            #             os.path.join(temp_checkpoint_dir, "model.pth")
            #         )
            #         checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            #     tune.report({"mean_accuracy": val_acc}, checkpoint=checkpoint)

            # save the train log to a file
            log_file = f"{self.outdir}/train_log.json"
            with open(log_file, "w") as f:
                json.dump(self.train_log, f, indent=4)

        self._verbose_printer(style="main_footer")

        return self.train_log
    
    def test(self):
        # set model to eval mode so we dont update the weights
        test_loss, test_acc = evaluate_model(self.model, self.test_loader, self.criterion, self.device)

        self.train_log['test_loss'].append(test_loss)
        self.train_log['test_acc'].append(test_acc)

        test_report = {
            "test_loss": test_loss,
            "test_acc": test_acc
        }

        self._verbose_printer(style="test_report", test_report=test_report)


        # load existing log file...if it exists
        # log_file = f"{self.outdir}/train_log.json"
        # try:
        #     with open(log_file, "r") as f:
        #         full_log = json.load(f)
        # except FileNotFoundError:
        #     full_log = {}
        
        # # append the test log to the full log
        # full_log['test_log'] = test_log

        # # save the full log to a file
        # with open(log_file, "w") as f:
        #     json.dump(full_log, f, indent=4)
            
        # return test_log


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
    model = BaselineCNN(num_classes=4, keep_prob=0.75)
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
    # train_log = fit(model, train_loader, val_loader, num_epochs=args.nepoch, 
    #                 optimizer=optimizer, criterion=criterion, device=device, 
    #                 lr_scheduler=lr_scheduler, outdir=args.outdir)
    #test_log = test(model, test_loader, criterion, device) # will save testing for later
