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

class CNN(nn.Module):
    def __init__(self, num_classes, keep_prob):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        # L1 ImgIn shape=(?, 256, 256, 3)
        #    Conv     -> (?, 256, 256, 32)
        #    Pool     -> (?, 128, 128, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 128, 128, 32)
        #    Conv      ->(?, 128, 128, 64)
        #    Pool      ->(?, 64, 64, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 64, 64, 64)
        #    Conv      ->(?, 64, 64, 128)
        #    Pool      ->(?, 32, 32, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=1 - keep_prob))
    
        # L4 Fully Connected Layer 128*32*32 inputs -> 256 outputs
        self.fc1 = nn.Sequential(
            nn.Linear(32*32*128, 256, bias=True),
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
    
if __name__ == "__main__":
    from dataloader import ImageDataset
    num_classes = 4
    model = CNN(num_classes=num_classes, keep_prob=0.75)
    print(model)
    model.summary()

    ##### Testing the dataloader #####
    csv_train_file = './data/train_info.csv'
    csv_test_file = './data/test_info.csv'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize all images to 256x256
        transforms.ToTensor()
    ])

    train_dataset = ImageDataset(csv_train_file, transform=transform)
    test_dataset = ImageDataset(csv_test_file, transform=transform)
    train_loader = ImageDataLoader(train_dataset).get_loader()
    test_loader = ImageDataLoader(test_dataset).get_loader()

    ##### Testing the model #####
    print("Training the model...")
    learning_rate = 0.001
    criterion = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(10):
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx+1} of {len(train_loader)}")
            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) # compute the loss
            optimizer.zero_grad() # reset the gradients for each epoch
            loss.backward() # backward pass
            optimizer.step() # update the weights

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    
    print("Training complete")
