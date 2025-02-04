import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from datetime import datetime

# wd = os.getcwd()
# print(wd)
class ImageDataset(Dataset):
  '''
  This class is used to load the image dataset from a csv file.
  '''
  def __init__(self, csvfilePath):
    self.csvfilePath = csvfilePath
    self.dataframe = pd.read_csv(csvfilePath)

    self.datasize = self.dataframe.shape[0]
    self.numLabel = self.dataframe['label_id'].nunique()
    self.labeldict = {idnum: self.dataframe.index[self.dataframe['label_id'] == idnum].to_list() for idnum in self.dataframe['label_id'].value_counts().index}
    self.transform = v2.Compose([
      v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
      v2.PILToTensor(),
      v2.ConvertImageDtype(torch.float32)
    ])

    dataLastModified = os.stat(self.csvfilePath).st_mtime
    self.dataLastModified = datetime.fromtimestamp(dataLastModified).strftime('%Y-%m-%d %H:%M:%S')
  
  def __len__(self):
    return self.datasize
  
  def __getitem__(self, idx):
    '''
    This function is used to get an item from the dataset.
    '''
    row = self.dataframe.iloc[idx] # get the row
    img = Image.open(row['image_path']).convert('RGB') # convert the image to RGB
    img = self.transform(img) # apply the transform to the image
    label = torch.tensor(row['label_id'], dtype = torch.long) # convert the label to a tensor
    return img, label
  
  def summary(self):
    '''
    prints a summary of the dataset
    '''
    print(f"\n{'='*40}")
    print(f"{' '*12}Dataset Summary")
    print(f"{'-'*40}")
    print(f"File Path:    {self.csvfilePath}")
    print(f"Last Updated: {self.dataLastModified}")
    print(f"{'-'*40}")
    print(f"Sample Sizes: {self.datasize}")
    print(f"Label Types: {self.numLabel}")
    for l in range(self.numLabel):
      print(f"Label {l}: {len(self.labeldict[l])} | {len(self.labeldict[l]) / self.datasize * 100:.2f}%")
    print(f"{'='*40}")




# dataInfo = pd.read_csv('./data/train_info.csv')
# dataInfo.reset_index()

# dataSize = dataInfo.shape[0] # total num of training data

# for idx, row in dataInfo.iterrows():
#   imgpath = row['image_path']
#   img = Image.open(imgpath)
#   imgArray = np.asarray(img)
#   print(imgArray.shape)
#   break


if __name__ == "__main__":
  testData = ImageDataset('./data/train_info.csv')
  print(testData.__getitem__(0)) # displays the first image and label as a tensor
  # print(testData[0])
  testData.summary()
  print(testData.labeldict[2][:10])
  print(testData[134][1])
  print(len(testData))