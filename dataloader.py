import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset
import os
class ImageDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4):
        """
        Initializes the DataLoader for the dataset.
        Args:
            dataset (ImageDataset): The dataset object.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data at every epoch.
            num_workers (int): Number of parallel workers for loading data.
        """
        # Maybe add a try except block to handle wrong batch size or num_workers values
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        # create a dataloader object using the data loader class from torch.utils.data
        self.dataloader = DataLoader(self.dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=self.shuffle, 
                                    num_workers=self.num_workers)

    def get_loader(self):
        return self.dataloader
    


if __name__ == "__main__":
    ##### testing the dataloader #####

    csv_file = './data/train_info.csv'
    dataset = ImageDataset(csv_file)
    print("testing dataloader")

    # test with default parameters
    print("\n1. testing with default parameters")
    train_loader = ImageDataLoader(dataset).get_loader()
    
    # check first batch
    data, target = next(iter(train_loader))
    print(f"Data size: {data.size(0)}")
    print(f"Data shape: {data.shape}")
    print(f"Batch size: {data.size(0)}")
    print(f"Image size: {data.size()[1:]}")
    print(f"Target size: {target.size()}")

    # checking with lower batch size
    print("\n2. testing with lower batch size = 8")
    small_batch_loader = ImageDataLoader(dataset, batch_size=8).get_loader()
    data, target = next(iter(small_batch_loader))
    print(f"Batch size: {data.size(0)}")
    print(f"Image size: {data.size()[1:]}")
    print(f"Target size: {target.size()}")

    # test data loading for multiple batches
    print("\n3. testing data loading for multiple batches")
    multi_batch_loader = ImageDataLoader(dataset, batch_size=8, shuffle=False).get_loader()
    for batch_idx, (data, target) in enumerate(multi_batch_loader):
        if batch_idx >= 3:
            break
        print(f"Batch {batch_idx + 1}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Target type: {target.dtype}")

    # test with shuffle = False
    print("\n4. testing with shuffle = False")
    no_shuffle_loader = ImageDataLoader(dataset, shuffle=False).get_loader()
    data, target = next(iter(no_shuffle_loader))
    print(f"Data shape: {data.shape}")
    print(f"Target shape: {target.shape}")

    # test for higher batch size
    print("\n5. testing for higher batch size = 64")
    large_batch_loader = ImageDataLoader(dataset, batch_size=64).get_loader()
    data, target = next(iter(large_batch_loader))
    print(f"Batch size: {data.size(0)}")
    print(f"Image size: {data.size()[1:]}")
    print(f"Target size: {target.size()}")