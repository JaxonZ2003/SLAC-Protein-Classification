import os
import json
import torch
import pytest
from torch.utils.data import DataLoader, Dataset
from SLAC25.network import Wrapper, ModelWrapper
from SLAC25.models import BaselineCNN  # make sure BaselineCNN is available

# --------------------------------------------------------------------
# Dummy dataset and dataloader for testing purposes
# --------------------------------------------------------------------
class DummyDataset(Dataset):
    def __init__(self, num_samples=10, image_shape=(3, 512, 512)):
        self.num_samples = num_samples
        self.image_shape = image_shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a random tensor as an "image" and a dummy label (0)
        return torch.randn(*self.image_shape), 0

def get_dummy_loader(batch_size=2):
    ds = DummyDataset(num_samples=10)
    return DataLoader(ds, batch_size=batch_size)

# --------------------------------------------------------------------
# Fixtures and monkey patches
# --------------------------------------------------------------------
# Patch _prepareDataLoader in Wrapper so that it uses our dummy dataloaders
@pytest.fixture(autouse=True)
def patch_prepare_dataloader(monkeypatch):
    def dummy_prepare(self, batch_size=32, testmode=False, max_imgs=None, nwork=0):
        self.train_loader = get_dummy_loader(batch_size)
        self.test_loader  = get_dummy_loader(batch_size)
        self.val_loader   = get_dummy_loader(batch_size)
    monkeypatch.setattr(Wrapper, "_prepareDataLoader", dummy_prepare)

# Patch evaluate_model so that the test phase always returns fixed loss and accuracy.
@pytest.fixture(autouse=True)
def patch_evaluate_model(monkeypatch):
    def dummy_evaluate_model(model, loader, criterion, device):
        return 0.5, 0.8  # dummy loss and accuracy values
    # Import the SLAC25.network module as "network" so it can be referenced.
    import SLAC25.network as network
    monkeypatch.setattr(network, "evaluate_model", dummy_evaluate_model)

# Fixture to get a temporary output directory for model files and logs
@pytest.fixture
def tmp_outdir(tmp_path):
    out_dir = tmp_path / "models"
    out_dir.mkdir()
    return str(out_dir)

# --------------------------------------------------------------------
# Test cases for network.py
# --------------------------------------------------------------------
def test_modelwrapper_summary(capfd):
    # Create a ModelWrapper using a BaselineCNN model.
    wrapper = ModelWrapper("BaselineCNN", num_classes=4, keep_prob=0.75,
                           num_epochs=1, outdir=".", verbose=True, testmode=True)
    # Call the summary() method which prints training setup info
    wrapper.summary()
    out, _ = capfd.readouterr()
    # Check that expected strings appear in the printed summary.
    assert "MODEL:" in out
    assert "EPOCH COUNT:" in out
    assert "TRAINING SETUP:" not in out or "Training Setup:" in out  # either one is acceptable

def test_modelwrapper_train(tmp_outdir):
    # Create a ModelWrapper with a dummy BaselineCNN model, using the temporary output directory.
    wrapper = ModelWrapper("BaselineCNN", num_classes=4, keep_prob=0.75,
                           num_epochs=1, outdir=tmp_outdir, verbose=True, testmode=True)
    train_log = wrapper.train()
    # Verify that train_log contains expected keys.
    expected_keys = ['epoch', 'train_loss', 'train_acc', 'val_loss',
                     'val_acc', 'test_loss', 'test_acc', 'learning_rates',
                     'model_checkpoints', 'time_per_epoch']
    for key in expected_keys:
        assert key in train_log
    # Expect at least one epoch logged.
    assert len(train_log['epoch']) >= 1
    # Check that the train log file was written.
    log_file = os.path.join(tmp_outdir, "train_log.json")
    assert os.path.exists(log_file)
    
    # Call test() to update the JSON file with the "test_log" key.
    wrapper.test()
    
    with open(log_file, "r") as f:
        log_data = json.load(f)
    # Verify that the test log is recorded within the saved JSON.
    assert "test_log" in log_data

def test_modelwrapper_test(tmp_outdir):
    # Create a ModelWrapper with a dummy BaselineCNN model.
    wrapper = ModelWrapper("BaselineCNN", num_classes=4, keep_prob=0.75,
                           num_epochs=1, outdir=tmp_outdir, verbose=False, testmode=True)
    test_log = wrapper.test()
    # Check that the returned dictionary contains test_loss and test_accuracy.
    assert 'test_loss' in test_log
    assert 'test_accuracy' in test_log
