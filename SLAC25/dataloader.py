import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler, SubsetRandomSampler
from SLAC25.sampler import StratifiedSampler
from SLAC25.dataset import ImageDataset
import os
# from torchvision import transforms
from PIL import Image


class DataLoaderFactory:
    def __init__(self, dataset, batch_size=32, num_workers=8, pin_memory=False, drop_last=False, shuffle=False):
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
        self.dataset = dataset
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.sampler = None
        self.shuffle = shuffle

        self.setBatchSize(batch_size)
        self.setNumWorkers(num_workers)

        # self.sampler = self._create_sampler(sampler_type, weights, indices)

        # self.dataloader = DataLoader(
        #     self.dataset,
        #     batch_size=self.batch_size,
        #     shuffle=self.shuffle if self.sampler is None else False,
        #     num_workers=self.num_workers,
        #     sampler=self.sampler
        # )
    
    def setRandomSampler(self, replacement=False, num_samples=None, generator=None):
        self.sampler = RandomSampler(self.dataset, replacement, num_samples, generator)
    
    def setWeightedRandomSampler(self, weights, num_samples=1000, replacement=True, generator=None):
        if weights != len(self.dataset):
             raise ValueError(
                 f"Invalid weights: Expected a list of length {len(self.dataset)}, "
                 f"but received a list of length {len(weights)}. Each data point in the dataset must have a corresponding weight."
        )

        self.sampler = WeightedRandomSampler(weights, num_samples, replacement, generator)
    
    def setSequentialSampler(self):
        self.sampler = SequentialSampler(self.dataset)
    
    def setSubsetRandomSampler(self, indices, generator=None, seed=1):
        if indices is None:
            raise ValueError("Indices are required for subset sampling.")
        
        self.sampler = SubsetRandomSampler(indices, generator)
    
    def setStratifiedSampler(self, samplePerGroup, allowRepeat=False):
        self.sampler = StratifiedSampler(data_source=self.dataset,
                                    samplePerGroup=samplePerGroup,
                                    allowRepeat=allowRepeat)       
    
    def setBatchSize(self, batch_size):
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        self.batch_size = batch_size
    
    def setNumWorkers(self, num_workers):
        if num_workers < 0:
            raise ValueError("Number of workers must be non-negative.")

        self.num_workers = num_workers

    def outputDataLoader(self):
        if self.sampler is None:
            raise RuntimeError("Please define a sampler for the dataLoader first before output a DataLoader")

        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          sampler=self.sampler,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,
                          shuffle=self.shuffle)

    # def _create_sampler(self, sampler_type, weights, indices):
    #     """
    #     Creates the appropriate sampler based on the sampler_type.

    #     Args:
    #         sampler_type (str): The type of sampler to use ('random', 'weighted', 'sequential', 'subset').
    #         weights (list, optional): Weights for WeightedRandomSampler.
    #         indices (list, optional): Indices for SubsetRandomSampler.

    #     Returns:
    #         torch.utils.data.Sampler or None: The configured sampler.
    #     """
    #     if sampler_type == 'random':
    #         return RandomSampler(self.dataset)
    #     elif sampler_type == 'weighted':
    #         if weights is None:
    #             raise ValueError("Weights are required for weighted sampling.")
    #         return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    #     elif sampler_type == 'sequential':
    #         return SequentialSampler(self.dataset)
    #     elif sampler_type == 'subset':
    #         if indices is None:
    #             raise ValueError("Indices are required for subset sampling.")
    #         return SubsetRandomSampler(indices)
    #     return None

    # def get_loader(self):
    #     return self.dataloader

if __name__ == "__main__":
    ##### Testing the dataloader #####
    package_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(package_root, "..", "data", "train_info.csv")
    data_path = os.path.abspath(data_path)
    #transform = transforms.Compose([
    #    transforms.Resize((256, 256)),  # Resize all images Shoto 256x256
    #    transforms.ToTensor()
    #])

    dataset = ImageDataset(data_path)
    dataloader = DataLoaderFactory(dataset, batch_size=3)
    print("Testing dataloader")

    # # Test with default parameters
    # print("\n1. Testing with default parameters")
    dataloader.setStratifiedSampler(1)
    train_loader = dataloader.outputDataLoader()
    for batch in train_loader:
        print(batch)
        break


    # data, target = next(iter(train_loader))
    # print(f"Data size: {data.size(0)}")
    # print(f"Data shape: {data.shape}")
    # print(f"Target shape: {target.size()}")

    # # Test with lower batch size
    # print("\n2. Testing with lower batch size = 8")
    # small_batch_loader = ImageDataLoader(dataset, batch_size=8).get_loader()
    # data, target = next(iter(small_batch_loader))
    # print(f"Batch size: {data.size(0)}")

    # # Test with weighted sampling
    # print("\n3. Testing with weighted sampling")
    # sample_weights = torch.ones(len(dataset))
    # weighted_loader = ImageDataLoader(dataset, batch_size=8, sampler_type='weighted', weights=sample_weights).get_loader()
    # data, target = next(iter(weighted_loader))
    # print(f"Batch size (Weighted Sampler): {data.size(0)}")

    # # Test with sequential sampling
    # print("\n4. Testing with sequential sampling")
    # sequential_loader = ImageDataLoader(dataset, batch_size=8, sampler_type='sequential').get_loader()
    # data, target = next(iter(sequential_loader))
    # print(f"Batch size (Sequential Sampler): {data.size(0)}")

    # # Test with subset sampling
    # print("\n5. Testing with subset sampling")
    # subset_indices = list(range(16))  # Selecting the first 16 samples
    # subset_loader = ImageDataLoader(dataset, batch_size=4, sampler_type='subset', indices=subset_indices).get_loader()
    # for batch_idx, (data, target) in enumerate(subset_loader):
    #     print(f"Batch {batch_idx + 1}:")
    #     print(f"  Data shape: {data.shape}")
    #     print(f"  Target shape: {target.shape}")

    # # Test for multiple batches
    # print("\n6. Testing data loading for multiple batches")
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     if batch_idx >= 3:
    #         break
    #     print(f"Batch {batch_idx + 1}:")
    #     print(f"  Data shape: {data.shape}")
    #     print(f"  Target shape: {target.shape}")

