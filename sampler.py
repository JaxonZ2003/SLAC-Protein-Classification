from torch.utils.data import Sampler
from typing import List

from dataset import ImageDataset

class BatchDownSampler(Sampler[List[int]]):
  def __init__(self, dataset, batchSize):
    self.data = dataset

    if not dataset.labeldict:
      self.labeldict = {idnum: self.dataframe.index[self.dataframe['label_id'] == idnum].to_list() for idnum in self.dataframe['label_id'].value_counts().index}
    self.labeldict = dataset.labeldict

    max_batch = min(len(v) for v in self.labeldict.values()) # can't exceed the num for the label with min samples
    if batchSize > max_batch:
      raise RuntimeError(f"The batch size must not exceed the number of samples in the label category with the fewest samples ({max_batch}).")
    self.batchSize = batchSize

  def __len__(self):
    return len(self.data) + (self.batchSize - 1) // self.batchSize # include extra partial batch
  
  def __iter__(self):
    pass


if __name__ == "__main__":
  test = ImageDataset('./data/train_info.csv')
  BatchDownSampler(test, 300000)
  print("test")