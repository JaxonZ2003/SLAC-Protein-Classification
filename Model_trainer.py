import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# model imports
from Models import CNN, ResNet

def fit(model, train_loader, val_loader, num_epochs, optimizer, criterion, device, save_every=5, callbacks=None, outdir='./models'):
    print(f'Starting training on {device}')
    # key can be the epoch number
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
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f'Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}')
        
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
            save_file = f"{outdir}/model_ep{epoch}.net"
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # simple data set of random data of values
    train_data = torch.randn(100, 3, 512, 512) # 100 images of 3 channels and 512x512 pixels
    train_labels = torch.randint(0, 4, (100,)) # 100 labels from 0 to 3

    # validation data same size as test data
    val_data = torch.randn(20, 3, 512, 512)
    val_labels = torch.randint(0, 4, (20,))

    # test data
    test_data = torch.randn(20, 3, 512, 512)
    test_labels = torch.randint(0, 4, (20,)) 

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=True)


    # simple model
    model = CNN(num_classes=4, keep_prob=0.75)
    model.to(device)

    criterion = nn.CrossEntropyLoss() # internally computes the softmax so no need for it. 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # train the model
    train_log = fit(model, train_loader, val_loader, num_epochs=5, optimizer=optimizer, criterion=criterion, device=device)
    test_log = test(model, test_loader, criterion, device)

    # plot the training and validation loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_log['train_loss_per_epoch'], label='Train Loss')
    plt.plot(train_log['val_loss_per_epoch'], label='Validation Loss')
    plt.legend()
    plt.show()

'''
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
    
    # add in a learning rate scheduler to reduce the learning rate by 10% every 10 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # add in a early stopping callback
    early_stopping = 
    # call backs 
    callbacks = [
        lr_scheduler
    ]

    train_log = train_model(model, train_loader, num_epochs=10, optimizer=optimizer, criterion=criterion, device=device, callbacks=callbacks)

    test_log = test_model(model, test_loader, criterion, device)
'''