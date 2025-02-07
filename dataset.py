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
  train_info = ImageDataset('./data/train_info.csv')
  print(train_info.__getitem__(0)) # displays the first image and label as a tensor
  # print(testData[0])
  # testData.summary()
  print(train_info.labeldict[2][:10])
  print(train_info[134][1])
  # print(len(testData))

  # TEST CASES
  # ensure data is a valid dataframe
  assert isinstance(train_info.dataframe, pd.DataFrame)

  # check column names
  expected_columns = ['image_path', 'image_id', 'label_id', 'label_text', 'label_raw', 'source']
  assert train_info.dataframe.columns.tolist() == expected_columns

  # verify shape and non-emptiness
  assert train_info.dataframe.shape[1] == 6
  assert not train_info.dataframe.empty

  # display DataFrame information
  print("DatafFrame Info:")
  print(train_info.dataframe.info())
  print("Random Sample Rows:")
  print(train_info.dataframe.sample(5))

  # print labels sorted by label_id
  sorted_labels = train_info.dataframe[['label_text', 'label_id']].drop_duplicates().sort_values(by='label_id')
  for index, row in sorted_labels.iterrows():
    print(f"Label: {row['label_text']}, Label ID: {row['label_id']}")

  # testing __len__()
  assert len(train_info) == train_info.dataframe.shape[0]
  print(f"Dataset Length: {len(train_info)}")

  # testing __getitem__()
  img, label = train_info[0]
  print(f"Image shape: {img.shape}, Label: {label}")
  ## check types and properties of returned tensors
  assert isinstance(img, torch.Tensor)
  assert isinstance(label, torch.Tensor)
  assert label.dtype == torch.long
  ## debug print statements for tensor shape and label
  print(f"Image Tensor Shape: {img.shape}")
  print(f"Label Tensor Shape: {label.shape}, Value: {label.item()}")
  print(f"Row at index 0:\n{train_info.dataframe.iloc[0]}")

  # testing summary()
  train_info.summary()
  assert train_info.datasize > 0
  assert isinstance(train_info.labeldict, dict)
  assert train_info.numLabel == len(train_info.labeldict)
