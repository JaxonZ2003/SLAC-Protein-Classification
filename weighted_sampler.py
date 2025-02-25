import random
import torch
from torch.utils.data import Sampler
import numpy as np
from dataset.dataset import ImageDataset

class WeightedRandomSampler(Sampler[int]):
    def __init__(self, dataset, weights: torch.Tensor, total_samples: int, allowRepeat=False):
        self.dataset = dataset
        self.weights = weights
        self.total_samples = total_samples
        self.allowRepeat = allowRepeat
        self.labeldict = dataset.labeldict

        if len(self.weights) != len(dataset):
            raise ValueError(f"Weights tensor must be of size {len(dataset)}")

    def __iter__(self):
        # normalize the weights so that their sum is 1
        normalized_weights = self.weights / self.weights.sum()
        selected_indices = []

        if self.allowRepeat:
            # sample with replacement
            sample = torch.multinomial(normalized_weights, self.total_samples, replacement=True)
        else:
            # sample without replacement
            sample = torch.multinomial(normalized_weights, self.total_samples, replacement=False)

        selected_indices.extend(sample.tolist())

        return iter(selected_indices)

    def __len__(self):
        return self.total_samples

def create_sample_weights(dataset):
    class_counts = {label: len(indices) for label, indices in dataset.labeldict.items()}

    total_samples = sum(class_counts.values())

    # create tensor to store the weights for each sample
    sample_weights = torch.zeros(total_samples)

    for label, indices in dataset.labeldict.items():
        # weight = inverse frequency
        weight = 1 / class_counts[label]
        sample_weights[indices] = weight

    return sample_weights

if __name__ == "__main__":
    dataset = ImageDataset('./data/train_info.csv')
    sample_weights = create_sample_weights(dataset)
    total_samples = 1000 

    weighted_sampler = WeightedRandomSampler(dataset, sample_weights, total_samples, allowRepeat=False)

    sampled_indices = list(iter(weighted_sampler))

    print(f"Sampled indices: {sampled_indices}")

    testdict = {}
    for idx in sampled_indices:
        label = dataset[idx][1].item()  
        if label not in testdict:
            testdict[label] = 1
        else:
            testdict[label] += 1
    
    print(f"Sampled label distribution: {testdict}")

    # TEST CASES
    assert len(sample_weights) == len(dataset)
    assert len(sampled_indices) == total_samples, f"Expected {total_samples} sampled indices, but got {len(sampled_indices)}"
