# This file seems to be redundant with sampler.py
import random
from torch.utils.data import Sampler
from typing import List

from dataset import ImageDataset

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


# === TEST CASES ===
if __name__ == "__main__":
    test = ImageDataset('./data/train_info.csv')
    
    print("== Testing Without Bootstrapping ==")
    a = EqualGroupSampler(test, 100, bootstrap=False)
    indices = list(iter(a))
    print(f"Sampled {len(indices)} indices:", indices[:10])  # Print first 10 samples

    print("== Testing With Bootstrapping ==")
    b = EqualGroupSampler(test, 100, bootstrap=True)
    indices_bootstrap = list(iter(b))
    print(f"Sampled {len(indices_bootstrap)} indices (bootstrapped):", indices_bootstrap[:10])  

    # Checking distribution of samples
    testdict = {}
    for i in indices_bootstrap:
        label = test[i][1].item()
        testdict[label] = testdict.get(label, 0) + 1

    print("Sample distribution:", testdict)
