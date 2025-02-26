from Model_trainer import *
from dataset import ImageDataset
from dataloader import ImageDataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *

'''
This script will be used to check the model if it works properly with no errors on our own device (i.e. not CSC).
We will select a few images from the train set and run the fit function to check if the model is working with no error.
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset_csv =  './data/train_info.csv'
train_dataset = pd.read_csv(train_dataset_csv)
# sample some images
train_dataset_sample = train_dataset.sample(n = 100)
# save to csv
train_dataset_sample.to_csv('./data/train_sample.csv', index=False)
# load the dataset
train_dataset = ImageDataset('./data/train_sample.csv')
# create a dataloader
train_loader = ImageDataLoader(train_dataset, batch_size=10, num_workers=4).get_loader()

# load model
model_check = (num_classes=4, keep_prob=0.75)
model_check.to(device)

# fit the model
train_log = fit(model_check, train_loader, num_epochs=10, optimizer=optim.Adam(model_check.parameters(), lr=0.001), criterion=nn.CrossEntropyLoss(), device=device, outdir='./data', lr_scheduler=None)

x = visualize_performance(train_log_path='./data/train_log.json', out_dir='./data', file_name='CNN_performance_plot.png')