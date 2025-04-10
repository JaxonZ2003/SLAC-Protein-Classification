import pytest
import torch
import torch.nn as nn
from SLAC25.models import BaselineCNN, ResNet

# Test the forward pass of BaselineCNN
def test_baselinecnn_forward():
    num_classes = 10
    keep_prob = 0.8
    model = BaselineCNN(num_classes=num_classes, keep_prob=keep_prob)
    
    # BaselineCNN expects an input of shape (batch, 3, 512, 512)
    x = torch.randn(2, 3, 512, 512)
    
    # Run the forward pass and check the output shape
    output = model(x)
    assert output.shape == (2, num_classes)

# Test the forward pass of ResNet (using resnet50 by default)
def test_resnet_forward():
    num_classes = 10
    keep_prob = 0.8
    model = ResNet(num_classes=num_classes, keep_prob=keep_prob, resnet_type='50')
    
    # ResNet expects an input of shape (batch, 3, 224, 224)
    x = torch.randn(2, 3, 224, 224)
    
    # Run the forward pass and check the output shape
    output = model(x)
    assert output.shape == (2, num_classes)

# Test that an invalid resnet type raises ValueError
def test_resnet_invalid_type():
    num_classes = 10
    keep_prob = 0.8
    with pytest.raises(ValueError):
        _ = ResNet(num_classes=num_classes, keep_prob=keep_prob, resnet_type='18')

# Test the transfer learning setup in ResNet
def test_resnet_transfer_learn():
    num_classes = 10
    keep_prob = 0.8
    model = ResNet(num_classes=num_classes, keep_prob=keep_prob, resnet_type='50')
    
    # Ensure all parameters in resnet are trainable before transfer learning is applied
    for param in model.resnet.parameters():
        assert param.requires_grad == True

    # Apply transfer learning: this should freeze all resnet parameters except those in the fc layer.
    model.transfer_learn()

    for name, param in model.resnet.named_parameters():
        if "fc" in name:
            # The fully connected layer's parameters should remain trainable
            assert param.requires_grad == True
        else:
            # All other parameters should be frozen
            assert param.requires_grad == False

    # Additionally, ensure that fc_layer1 and fc_layer2 are trainable
    for param in model.fc_layer1.parameters():
        assert param.requires_grad == True
    for param in model.fc_layer2.parameters():
        assert param.requires_grad == True

# Test that BaselineCNN.summary() prints model information
def test_baselinecnn_summary(capfd):
    num_classes = 10
    keep_prob = 0.8
    model = BaselineCNN(num_classes=num_classes, keep_prob=keep_prob)
    
    model.summary()  # This prints the model summary to stdout
    
    out, err = capfd.readouterr()
    # Check for the model class name in the summary output
    assert "BaselineCNN" in out

# Test that ResNet.print_model_summary() prints a summary for the resnet part
def test_resnet_print_model_summary(capfd):
    num_classes = 10
    keep_prob = 0.8
    model = ResNet(num_classes=num_classes, keep_prob=keep_prob, resnet_type='50')
    
    model.print_model_summary()  # This prints the summary of the resnet portion
    
    out, err = capfd.readouterr()
    # The output from torchsummary often contains total parameter counts; check for a common substring
    assert "Total params:" in out or "Parameters:" in out
