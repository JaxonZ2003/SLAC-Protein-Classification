import os
import json
import pytest
import pandas as pd
import torch
from PIL import Image
from datetime import datetime
from SLAC25.dataset import ImageDataset

# ----------------------------------------------------------------------------
# Fixture to create a temporary CSV file and dummy images.
# ----------------------------------------------------------------------------
@pytest.fixture
def dummy_csv(tmp_path):
    # Create a directory for dummy images.
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    data = []
    # Create 3 dummy images.
    for i in range(3):
        img_path = images_dir / f"image_{i}.png"
        # Create a simple 100x100 RGB image (e.g. shades of gray).
        img = Image.new("RGB", (100, 100), color=(i * 40, i * 40, i * 40))
        img.save(img_path)
        # For demonstration, set label_id as 0 for even i and 1 for odd i.
        data.append({"image_path": str(img_path), "label_id": i % 2})
    df = pd.DataFrame(data)
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

# ----------------------------------------------------------------------------
# Fixture to monkey-patch ImageDataset._loadConfig so that it returns a dummy config.
# ----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_load_config(monkeypatch):
    dummy_config = {
       "visualizeAndSave": {
         "title_before_transform": {"x": 0.5, "y": 0.95, "s": "Before Transform"},
         "title_after_transform": {"x": 0.5, "y": 0.95, "s": "After Transform"},
         "label_text_params": {"x": 0.1, "y": 0.1, "s": "Label: {}"},
         "label_transform_logs": {"x": 0.1, "y": 0.05, "s": "Logs: {}"}
       }
    }
    monkeypatch.setattr("SLAC25.dataset.ImageDataset._loadConfig", lambda self: dummy_config)

# ----------------------------------------------------------------------------
# Test the __len__ method: number of rows in the CSV should be the dataset size.
# ----------------------------------------------------------------------------
def test_dataset_len(dummy_csv):
    dataset = ImageDataset(dummy_csv, transform=None, config=None, recordTransform=True)
    assert len(dataset) == 3

# ----------------------------------------------------------------------------
# Test the getImagePath and getLabelId methods.
# ----------------------------------------------------------------------------
def test_get_image_path_and_label(dummy_csv):
    dataset = ImageDataset(dummy_csv, transform=None, config=None, recordTransform=True)
    img_path = dataset.getImagePath(0)
    label = dataset.getLabelId(0)
    df = pd.read_csv(dummy_csv)
    assert img_path == df.iloc[0]['image_path']
    assert label == df.iloc[0]['label_id']

# ----------------------------------------------------------------------------
# Test that __getitem__ returns a tensor image (with appropriate shape) and a label tensor.
# ----------------------------------------------------------------------------
def test_getitem_returns_tensor_and_label(dummy_csv):
    dataset = ImageDataset(dummy_csv, transform=None, config=None, recordTransform=True)
    # __getitem__ calls the TransformV1 preprocessing, which resizes images to 512x512
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    # Check that the image tensor is 3 x 512 x 512
    assert img.shape == (3, 512, 512)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long

# ----------------------------------------------------------------------------
# Test the summary method output.
# ----------------------------------------------------------------------------
def test_summary_output(dummy_csv, capsys):
    dataset = ImageDataset(dummy_csv, transform=None, config=None, recordTransform=True)
    dataset.summary()
    captured = capsys.readouterr().out
    # Check that key strings are printed.
    assert "Dataset Summary" in captured
    assert "File Path:" in captured
    assert "Sample Sizes:" in captured

# ----------------------------------------------------------------------------
# Test the visualizeAndSave method.
# ----------------------------------------------------------------------------
def test_visualize_and_save(dummy_csv, tmp_path):
    dataset = ImageDataset(dummy_csv, transform=None, config=None, recordTransform=True)
    # Use a temporary directory for saving images.
    save_dir = tmp_path / "output_images"
    save_dir.mkdir()
    # Call visualizeAndSave for an index (e.g., index 1).
    dataset.visualizeAndSave(1, savedPath=str(save_dir))
    # The file name pattern: "{datasetType}_{timeNow}_{idx}.png" is used.
    # Since our CSV filename is "dummy.csv", _checkTrainTest should return "Others".
    # We then search for a file ending with "_1.png"
    files = list(save_dir.glob("*_1.png"))
    assert len(files) == 1
