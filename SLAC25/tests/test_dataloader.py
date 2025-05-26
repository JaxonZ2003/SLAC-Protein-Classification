import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from SLAC25.dataloader import DataLoaderFactory
from SLAC25.dataset import ImageDataset

# Create a temporary directory for testing
TEST_DIR = "/tmp/test_dataloader"
os.makedirs(TEST_DIR, exist_ok=True)

# Generate 10 fake images and labels
def create_fake_data():
    image_dir = os.path.join(TEST_DIR, "images")
    os.makedirs(image_dir, exist_ok=True)
    labels = []
    for i in range(10):
        # Create a fake image
        img = Image.new("RGB", (64, 64), color=(i * 25, i * 25, i * 25))
        img_path = os.path.join(image_dir, f"image_{i}.png")
        img.save(img_path)
        # labels
        labels.append({"image_path": img_path, "label_id": i % 4})  

    # Save labels to a CSV file
    labels_df = pd.DataFrame(labels)
    labels_csv_path = os.path.join(TEST_DIR, "labels.csv")
    labels_df.to_csv(labels_csv_path, index=False)
    return labels_csv_path

# Mock training loop
def mock_training_loop(dataloader):
    for batch in dataloader:  
        images, labels = batch
        assert images.shape[0] > 0  # Ensure batch has data
        assert labels.shape[0] > 0
    print("Training loop executed successfully.")

# Test for random sampler
def test_random_sampler():
    labels_csv_path = create_fake_data()
    dataset = ImageDataset(labels_csv_path)
    factory = DataLoaderFactory(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    factory.setRandomSampler()
    dataloader = factory.outputDataLoader()
    mock_training_loop(dataloader)
    print("Random sampler test passed successfully.")

# Test for sequential sampler
def test_sequential_sampler():
    labels_csv_path = create_fake_data()
    dataset = ImageDataset(labels_csv_path)
    factory = DataLoaderFactory(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    factory.setSequentialSampler()
    dataloader = factory.outputDataLoader()
    mock_training_loop(dataloader)
    print("Sequential sampler test passed successfully.")

# Test for subset random sampler
def test_subset_random_sampler():
    labels_csv_path = create_fake_data()
    dataset = ImageDataset(labels_csv_path)
    factory = DataLoaderFactory(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    # Use a subset of indices (e.g., first 5 samples)
    indices = list(range(5))
    factory.setSubsetRandomSampler(indices)
    dataloader = factory.outputDataLoader()
    mock_training_loop(dataloader)
    print("Subset random sampler test passed successfully.")

# Test for weighted sampler
def test_weighted_sampler():
    labels_csv_path = create_fake_data()
    dataset = ImageDataset(labels_csv_path)
    factory = DataLoaderFactory(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    # Create a uniform weights list for each data point in the dataset
    weights = [1.0] * len(dataset)
    factory.setWeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    dataloader = factory.outputDataLoader()
    mock_training_loop(dataloader)
    print("Weighted sampler test passed successfully.")

# Test for stratified sampler
def test_stratified_sampler():
    labels_csv_path = create_fake_data()
    dataset = ImageDataset(labels_csv_path)
    factory = DataLoaderFactory(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    # Sample 1 item per label group; the group is determined by 'label_id'
    factory.setStratifiedSampler(samplePerGroup=1)
    dataloader = factory.outputDataLoader()
    mock_training_loop(dataloader)
    print("Stratified sampler test passed successfully.")


if __name__ == "__main__":
    test_random_sampler()
    test_sequential_sampler()
    test_subset_random_sampler()
    test_weighted_sampler()
    test_stratified_sampler()