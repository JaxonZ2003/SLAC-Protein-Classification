import random

from torch.utils.data import Sampler
from typing import List

from dataset.dataset import ImageDataset

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
    

if __name__ == "__main__":
  test = ImageDataset('./data/train_info.csv')
  a = StratifiedSampler(test, 23299)
  print(len(a) == 23299 * 4)

  b = StratifiedSampler(test, 100)
  indices = list(iter(b))
  print(indices)

  testdict = {}
  for i in indices:
    label = test[i][1].item()
    if label not in testdict.keys():
      testdict[label] = 1
    
    else:
      testdict[label] +=1

  print(testdict)
  print(len(b) == len(indices))


  # TEST CASES
  # testing __len__()
  expected_len = sum(min(len(v), a.samplePerGroup) for v in a.labeldict.values())
  assert len(a) == expected_len, f"Expected length: {expected_len}, but got: {len(test)}"
 