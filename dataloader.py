import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler, SubsetRandomSampler
from dataset import ImageDataset
import os
from torchvision import transforms
from PIL import Image


class ImageDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=10, sampler_type=None, weights=None, indices=None):
        """
        Initializes the DataLoader for the dataset.

        Args:
            dataset (ImageDataset): The dataset object.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data at every epoch.
            num_workers (int): Number of parallel workers for loading data.
            sampler_type (str, optional): Type of sampler ('random', 'weighted', 'sequential', 'subset').
            weights (list, optional): Sample weights if using weighted sampling.
            indices (list, optional): List of indices for subset sampling.
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")
        if num_workers < 0:
            raise ValueError("Number of workers must be non-negative.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sampler = self._create_sampler(sampler_type, weights, indices)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if self.sampler is None else False,
            num_workers=self.num_workers,
            sampler=self.sampler
        )

    def _create_sampler(self, sampler_type, weights, indices):
        """
        Creates the appropriate sampler based on the sampler_type.

        Args:
            sampler_type (str): The type of sampler to use ('random', 'weighted', 'sequential', 'subset').
            weights (list, optional): Weights for WeightedRandomSampler.
            indices (list, optional): Indices for SubsetRandomSampler.

        Returns:
            torch.utils.data.Sampler or None: The configured sampler.
        """
        if sampler_type == 'random':
            return RandomSampler(self.dataset)
        elif sampler_type == 'weighted':
            if weights is None:
                raise ValueError("Weights are required for weighted sampling.")
            return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        elif sampler_type == 'sequential':
            return SequentialSampler(self.dataset)
        elif sampler_type == 'subset':
            if indices is None:
                raise ValueError("Indices are required for subset sampling.")
            return SubsetRandomSampler(indices)
        return None

    def get_loader(self):
        return self.dataloader

if __name__ == "__main__":
    ##### Testing the dataloader #####
    csv_file = './data/train_info.csv'
    #transform = transforms.Compose([
    #    transforms.Resize((256, 256)),  # Resize all images to 256x256
    #    transforms.ToTensor()
    #])

    dataset = ImageDataset(csv_file)
    print("Testing dataloader")

    # Test with default parameters
    print("\n1. Testing with default parameters")
    train_loader = ImageDataLoader(dataset).get_loader()

    data, target = next(iter(train_loader))
    print(f"Data size: {data.size(0)}")
    print(f"Data shape: {data.shape}")
    print(f"Target shape: {target.size()}")

    # Test with lower batch size
    print("\n2. Testing with lower batch size = 8")
    small_batch_loader = ImageDataLoader(dataset, batch_size=8).get_loader()
    data, target = next(iter(small_batch_loader))
    print(f"Batch size: {data.size(0)}")

    # Test with weighted sampling
    print("\n3. Testing with weighted sampling")
    sample_weights = torch.ones(len(dataset))
    weighted_loader = ImageDataLoader(dataset, batch_size=8, sampler_type='weighted', weights=sample_weights).get_loader()
    data, target = next(iter(weighted_loader))
    print(f"Batch size (Weighted Sampler): {data.size(0)}")

    # Test with sequential sampling
    print("\n4. Testing with sequential sampling")
    sequential_loader = ImageDataLoader(dataset, batch_size=8, sampler_type='sequential').get_loader()
    data, target = next(iter(sequential_loader))
    print(f"Batch size (Sequential Sampler): {data.size(0)}")

    # Test with subset sampling
    print("\n5. Testing with subset sampling")
    subset_indices = list(range(16))  # Selecting the first 16 samples
    subset_loader = ImageDataLoader(dataset, batch_size=4, sampler_type='subset', indices=subset_indices).get_loader()
    for batch_idx, (data, target) in enumerate(subset_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Target shape: {target.shape}")

    # Test for multiple batches
    print("\n6. Testing data loading for multiple batches")
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 3:
            break
        print(f"Batch {batch_idx + 1}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Target shape: {target.shape}")

