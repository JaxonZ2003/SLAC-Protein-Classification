import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from dataloader import ImageDataLoader

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

    # Update ImageDataset to support transforms
    class ImageDataset:
        def __init__(self, csv_file, transform=None):
            self.data_frame = pd.read_csv(csv_file)
            self.transform = transform

        def __len__(self):
            return len(self.data_frame)

        def __getitem__(self, idx):
            img_path = self.data_frame.iloc[idx, 0]
            label = self.data_frame.iloc[idx, 1]

            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label)

    train_dataset = ImageDataset(csv_train_file, transform=transform)
    test_dataset = ImageDataset(csv_test_file, transform=transform)
    train_loader = ImageDataLoader(train_dataset).get_loader()
    test_loader = ImageDataLoader(test_dataset).get_loader()

    
