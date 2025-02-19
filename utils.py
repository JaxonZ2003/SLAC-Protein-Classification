# class for learning rate scheduler and early stopping, etc.
import torch

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


class LRScheduler:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
    def __call__(self, epoch, metrics):
        if self.mode == 'min':
            if metrics < self.best_loss:
                self.best_loss = metrics
                self.counter = 0
            elif metrics > self.best_loss:
                