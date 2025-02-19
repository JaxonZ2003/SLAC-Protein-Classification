import torch
import torch.nn as nn
import torchvision.models as models

######################### Simple CNN Model ###############################

class CNN(nn.Module):
    def __init__(self, num_classes, keep_prob):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.keep_prob = keep_prob
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
    
        # L4 Fully Connected Layer 128*input_size//8*input_size//8 inputs -> 256 outputs
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 64 * 64, 256, bias=True),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(p=1 - keep_prob)
        )
        # L5 Fully Connected Layer 256 inputs -> num_classes outputs
        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes, bias=True) # no need for softmax since the loss function is cross entropy
        )

    def forward(self, x):
        out = self.layer1(x) # Conv -> ReLU -> MaxPool -> Dropout
        out = self.layer2(out) # Conv -> ReLU -> MaxPool -> Dropout
        out = self.layer3(out) # Conv -> ReLU -> MaxPool -> Dropout
        out = out.view(out.size(0), -1) # Flatten them for FC, should be
        out = self.fc1(out) # FC -> ReLU -> Dropout
        out = self.fc2(out) # FC -> logits for our criterion
        return out

    def summary(self):
        print(self)

######################### ResNet50 Model ###############################

class ResNet(nn.Module):
    def __init__(self, num_classes, keep_prob):
        super(ResNet, self).__init__()
        # load the pretrained resnet50 model
        self.resnet = models.resnet50(pretrained=True)

        # freeze all layers but the last one
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # remove the last fc layer add our custom fc layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=1 - keep_prob),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)
