import random
import torch
from torch.utils.data import Sampler
import numpy as np
from dataset import ImageDataset

class WeightedRandomSampler(Sampler[int]):
    def __init__(self, dataset, weights: torch.Tensor, samplePerGroup, allowRepeat=False):
        self.dataset = dataset
        self.weights = weights
        self.samplePerGroup = samplePerGroup
        self.allowRepeat = allowRepeat
        self.labeldict = dataset.labeldict

        # Ensure the weights tensor is the same size as the dataset
        if len(weights) != len(dataset):
            raise ValueError(f"Weights tensor must be of size {len(dataset)}")

    def __iter__(self):
        labeldict_copy = {group: indices[:] for group, indices in self.labeldict.items()}
        selected_indices = []

        for group, indices in labeldict_copy.items():
            # create weights specific to the indices of each label
            group_weights = self.weights[indices]
            
            # normalize the group weights to sum to 1
            normalized_group_weights = group_weights / group_weights.sum()

            # sample indices for this group
            if len(indices) > self.samplePerGroup:
                # sample with replacement 
                if self.allowRepeat:
                    sample = torch.multinomial(normalized_group_weights, self.samplePerGroup, replacement=True)
                # sample without replacement
                else:
                    sample = torch.multinomial(normalized_group_weights, self.samplePerGroup, replacement=False)
                selected_indices.extend([indices[i] for i in sample])
            else:
                # if fewer samples than required, sample all
                selected_indices.extend(indices)
        
        # shuffle the selected indices if repetition is not allowed
        if not self.allowRepeat:
            random.shuffle(selected_indices)

        return iter(selected_indices)

    def __len__(self):
        total_samples = 0
        for group in self.labeldict.values():
            total_samples += min(len(group), self.samplePerGroup)
        return total_samples

def create_sample_weights(dataset):
    # calculate class frequencies
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
    weighted_sampler = WeightedRandomSampler(dataset, sample_weights, samplePerGroup=100)

    sampled_indices = list(iter(weighted_sampler))
    print(sampled_indices)

    testdict = {}
    for idx in sampled_indices:
        label = dataset[idx][1].item()  
        if label not in testdict:
            testdict[label] = 1
        else:
            testdict[label] += 1
    
    print(testdict)

    # TEST CASES
    assert len(sample_weights) == len(dataset)
    expected_samples = weighted_sampler.samplePerGroup * len(weighted_sampler.labeldict)
    assert len(sampled_indices) == expected_samples, f"Expected {expected_samples} sampled indices, but got {len(sampled_indices)}"