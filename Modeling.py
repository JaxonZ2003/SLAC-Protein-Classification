import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from dataloader import ImageDataLoader
from dataset import ImageDataset
from dataloader import ImageDataset

class CNN(nn.Module):
    def __init__(self, num_classes, keep_prob, input_size):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.input_size = input_size
        # L1 ImgIn shape=(?, input_size, input_size, 3)
        #    Conv     -> (?, input_size, input_size, 32)
        #    Pool     -> (?, input_size//2, input_size//2, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, input_size//2, input_size//2, 32)
        #    Conv      ->(?, input_size//2, input_size//2, 64)
        #    Pool      ->(?, input_size//4, input_size//4, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, input_size//4, input_size//4, 64)
        #    Conv      ->(?, input_size//4, input_size//4, 128)
        #    Pool      ->(?, input_size//8, input_size//8, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=1 - keep_prob))
    
        # L4 Fully Connected Layer 128*input_size*input_size inputs -> 256 outputs
        self.fc1 = nn.Sequential(
            nn.Linear((self.input_size//8)*(self.input_size//8)*128, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=1 - keep_prob)
        )
        # L5 Fully Connected Layer 256 inputs -> num_classes outputs
        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.layer1(x) # Conv -> ReLU -> MaxPool -> Dropout
        out = self.layer2(out) # Conv -> ReLU -> MaxPool -> Dropout
        out = self.layer3(out) # Conv -> ReLU -> MaxPool -> Dropout
        out = out.view(out.size(0), -1) # Flatten them for FC
        out = self.fc1(out) # FC -> ReLU -> Dropout
        out = self.fc2(out) # FC -> Softmax
        return out

    def summary(self):
        print(self)
    
def train_model(model, train_loader, num_epochs, optimizer, criterion, device):
    print(f'Starting training on {device}')
    train_log = {
        'train_loss_per_epoch': [],
        'train_acc_per_epoch': []
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
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f'Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}')
        
        # Calculate epoch stats
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_log['train_loss_per_epoch'].append(epoch_loss)
        train_log['train_acc_per_epoch'].append(epoch_acc)
        
        print(f'Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

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

def test_model(model, dataloader, criterion, device):
    model.eval()
    test_loss, test_acc = evaluate_model(model, dataloader, criterion, device)
    
    test_log = {
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    return test_log

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset paths
    csv_train_file = './data/train_info.csv'
    csv_test_file = './data/test_info.csv'
    
    # load the datasets
    train_dataset = ImageDataset(csv_train_file)
    test_dataset = ImageDataset(csv_test_file)
    train_loader = ImageDataLoader(train_dataset).get_loader()
    test_loader = ImageDataLoader(test_dataset).get_loader()

    # load the model
    model = CNN(num_classes=4, keep_prob=0.75, input_size=512)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_log = train_model(model, train_loader, num_epochs=10, optimizer=optimizer, criterion=criterion, device=device)

    test_log = test_model(model, test_loader, criterion, device)
