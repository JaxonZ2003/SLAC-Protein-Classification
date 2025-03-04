import numpy as np
import random
import os
import torch
from torch.utils.data import Sampler
from SLAC25.dataset import ImageDataset
from typing import List


class StratifiedSampler(Sampler):
  def __init__(self, data_source, samplePerGroup, allowRepeat=False):
    self.data_source = data_source
    self.allowRepeat = allowRepeat
    self.labeldict = data_source.labeldict

    if not self.allowRepeat:
      maxSamplePerGroup = min(len(v) for v in self.labeldict.values()) # can't exceed the num for the label with min samples

      if samplePerGroup > maxSamplePerGroup:
        raise RuntimeError(f"The sample size per group must not exceed the number of samples in the label category with the fewest samples ({maxSamplePerGroup}).")
    
    self.samplePerGroup = samplePerGroup
    # if not dataset.labeldict:
    #   self.labeldict = {idnum: self.dataframe.index[self.dataframe['label_id'] == idnum].to_list() for idnum in self.dataframe['label_id'].value_counts().index}

  def __len__(self):
    total = 0
    for indices in self.labeldict.values():
      total += min(len(indices), self.samplePerGroup)

    return total
  
  def __iter__(self):
    labeldict_copy = {group: indices[:] for group, indices in self.labeldict.items()}
    selected_indices = []
    for group, indices in labeldict_copy.items(): # iterate num groups times
      if len(indices) > self.samplePerGroup:
        sample = random.sample(indices, self.samplePerGroup)
      
      else:
        sample = indices[:] # if exceeding num of elements in the group
      
      if not self.allowRepeat:
          labeldict_copy[group] = [i for i in labeldict_copy[group] if i not in sample]

      selected_indices.extend(sample)
    
    random.shuffle(selected_indices)
    return iter(selected_indices)

  

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


class EqualGroupSampler(Sampler[int]):
    def __init__(self, dataset, samplePerGroup, allowRepeat=False, bootstrap=False):
        self.dataset = dataset
        self.allowRepeat = allowRepeat
        self.bootstrap = bootstrap
        self.labeldict = dataset.labeldict

        if not self.allowRepeat and not self.bootstrap:
            maxSamplePerGroup = min(len(v) for v in self.labeldict.values())  # Can't exceed the smallest label category
            
            if samplePerGroup > maxSamplePerGroup:
                raise RuntimeError(f"The sample size per group must not exceed the number of samples in the label category with the fewest samples ({maxSamplePerGroup}).")

        self.samplePerGroup = samplePerGroup

    def __len__(self):
        """Returns the total number of samples to be drawn."""
        total = 0
        for indices in self.labeldict.values():
            total += self.samplePerGroup  # Bootstrap allows sampling beyond actual size
        return total

    def __iter__(self):
        """Generate indices with bootstrapping (sampling with replacement)."""
        selected_indices = []

        for group, indices in self.labeldict.items():  
            if self.bootstrap:
                sample = random.choices(indices, k=self.samplePerGroup)  # Sampling with replacement
            elif len(indices) > self.samplePerGroup:
                sample = random.sample(indices, self.samplePerGroup)  # Sampling without replacement
            else:
                sample = indices[:]  # If fewer than samplePerGroup, take all
            
            selected_indices.extend(sample)
        
        random.shuffle(selected_indices)  # Shuffle the final sampled list
        return iter(selected_indices)
    

if __name__ == "__main__":
  package_root = os.path.dirname(os.path.abspath(__file__))
  data_path = os.path.join(package_root, "..", "data", "train_info.csv")
  data_path = os.path.abspath(data_path)
  test_dataset = ImageDataset(data_path)

  # Test StratifiedSampler
  print("\n=== Testing StratifiedSampler ===")
  stratified_sampler = StratifiedSampler(test_dataset, 100)
  indices = list(iter(stratified_sampler))
  
  # Check distribution across labels
  label_dist = {}
  for i in indices:
    label = test_dataset[i][1].item()
    label_dist[label] = label_dist.get(label, 0) + 1
  print("Label distribution:", label_dist)
  print(f"Total samples: {len(indices)}")
  assert len(stratified_sampler) == len(indices), "Length mismatch between sampler and sampled indices"

  # Test WeightedRandomSampler
  print("\n=== Testing WeightedRandomSampler ===")
  sample_weights = create_sample_weights(test_dataset)
  total_samples = 1000
  weighted_sampler = WeightedRandomSampler(test_dataset, sample_weights, total_samples, allowRepeat=False)
  weighted_indices = list(iter(weighted_sampler))

  # Check weighted sampling distribution
  weighted_dist = {}
  for idx in weighted_indices:
      label = test_dataset[idx][1].item()
      weighted_dist[label] = weighted_dist.get(label, 0) + 1
  print("Weighted sampling distribution:", weighted_dist)
  print(f"Total weighted samples: {len(weighted_indices)}")
  
  # Basic assertions
  assert len(sample_weights) == len(test_dataset), "Sample weights length mismatch"
  assert len(weighted_indices) == total_samples, "Incorrect number of weighted samples"

  # Test EqualGroupSampler
  print("\n=== Testing EqualGroupSampler ===")
  
  # Test without bootstrapping
  print("-- Without Bootstrapping --")
  equal_sampler = EqualGroupSampler(test_dataset, 100, bootstrap=False)
  equal_indices = list(iter(equal_sampler))
  print(f"Number of samples: {len(equal_indices)}")
  print("First 10 indices:", equal_indices[:10])

  # Test with bootstrapping
  print("\n-- With Bootstrapping --")
  bootstrap_sampler = EqualGroupSampler(test_dataset, 100, bootstrap=True)
  bootstrap_indices = list(iter(bootstrap_sampler))
  
  # Check bootstrap distribution
  bootstrap_dist = {}
  for i in bootstrap_indices:
      label = test_dataset[i][1].item()
      bootstrap_dist[label] = bootstrap_dist.get(label, 0) + 1
  print("Bootstrap distribution:", bootstrap_dist)
  print(f"Total bootstrap samples: {len(bootstrap_indices)}")