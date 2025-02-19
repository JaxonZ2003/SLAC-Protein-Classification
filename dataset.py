import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from datetime import datetime
import random

# wd = os.getcwd()
# print(wd)
class ImageDataset(Dataset):
  def __init__(self, csvfilePath):
    self.csvfilePath = csvfilePath
    self.dataframe = pd.read_csv(csvfilePath)
    self.train = True
    self.datasize = self.dataframe.shape[0]
    self.numLabel = self.dataframe['label_id'].nunique()
    self.labeldict = {idnum: self.dataframe.index[self.dataframe['label_id'] == idnum].to_list() for idnum in self.dataframe['label_id'].value_counts().index}
    self.train_transform = v2.Compose([
      v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
      v2.PILToTensor(),
      v2.ConvertImageDtype(torch.float32)
    ])
    self.val_transform = v2.Compose([
      v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
      v2.PILToTensor(),
      v2.ConvertImageDtype(torch.float32)
    ])

    dataLastModified = os.stat(self.csvfilePath).st_mtime
    self.dataLastModified = datetime.fromtimestamp(dataLastModified).strftime('%Y-%m-%d %H:%M:%S')
  
  def __len__(self):
    return self.datasize
  
  def __getitem__(self, idx):
    row = self.dataframe.iloc[idx] # get the row
    img = Image.open(row['image_path'])
    
    # check if the image is 3 channels
    if img.mode != 'RGB':
      img = img.convert('RGB')
    img = self.train_transform(img) if self.train else self.val_transform(img)

    # apply augmentations
    img = self.random_rotation(img) if self.train else img
    img = self.random_horizontal_flip(img) if self.train else img
    img = self.random_vertical_flip(img) if self.train else img
    img = self.random_gaussian_blur(img) if self.train else img

    label = torch.tensor(row['label_id'], dtype = torch.long) # convert the label to a tensor
    return img, label

  def random_gaussian_blur(self, img):
    """Gaussian blur with 50% probability"""
    if random.random() < 0.5:
        print("Applying Gaussian Blur")
        return v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(img)
    return img

  def random_rotation(self, img):
    """Random rotation between -45 to 45 degrees with a 50% probability"""
    if random.random() < 0.5:
        angle = random.uniform(-45, 45)
        print(f"Rotating by {angle:.2f} degrees")
        return v2.RandomRotation(degrees=(angle, angle))(img)
    return img

  def random_horizontal_flip(self, img):
    """Horizontal flip with 50% probability"""
    if random.random() < 0.5:
        print("Applying Horizontal Flip")
        return v2.RandomHorizontalFlip()(img)
    return img

  def random_vertical_flip(self, img):
    """Vertical flip with 50% probability"""
    if random.random() < 0.5:
        print("Applying Vertical Flip")
        return v2.RandomVerticalFlip()(img)
    return img

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
  # testData.summary()
  print(testData.labeldict[2][:10])
  print(testData[134][1])
  # print(len(testData))

  # TEST CASES
  # ensure data is a valid dataframe
  assert isinstance(testData.dataframe, pd.DataFrame)

  # check column names
  expected_columns = ['image_path', 'image_id', 'label_id', 'label_text', 'label_raw', 'source']
  assert testData.dataframe.columns.tolist() == expected_columns

  # verify shape and non-emptiness
  assert testData.dataframe.shape[1] == 6
  assert not testData.dataframe.empty

  # display DataFrame information
  print("DatafFrame Info:")
  print(testData.dataframe.info())
  print("Random Sample Rows:")
  print(testData.dataframe.sample(5))

  # print labels sorted by label_id
  sorted_labels = testData.dataframe[['label_text', 'label_id']].drop_duplicates().sort_values(by='label_id')
  for index, row in sorted_labels.iterrows():
    print(f"Label: {row['label_text']}, Label ID: {row['label_id']}")

  # testing __len__()
  assert len(testData) == testData.dataframe.shape[0]
  print(f"Dataset Length: {len(testData)}")

  # testing __getitem__()
  img, label = testData[0]
  print(f"Image shape: {img.shape}, Label: {label}")
  ## check types and properties of returned tensors
  assert isinstance(img, torch.Tensor)
  assert isinstance(label, torch.Tensor)
  assert label.dtype == torch.long
  ## debug print statements for tensor shape and label
  print(f"Image Tensor Shape: {img.shape}")
  print(f"Label Tensor Shape: {label.shape}, Value: {label.item()}")
  print(f"Row at index 0:\n{testData.dataframe.iloc[0]}")
  ## testing augmentation transformations
  for i in range(10):
    print(f"\nSample {i}:")
    img, label = testData[i]
  

  # testing summary()
  testData.summary()
  assert testData.datasize > 0
  assert isinstance(testData.labeldict, dict)
  assert testData.numLabel == len(testData.labeldict)
