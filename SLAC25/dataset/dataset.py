import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pytz
import json
import random
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from datetime import datetime

transformConfig = {
    "rotation": {
        "on": True,
        "params": {
            "degrees": (-45, 45)
        }
    },
    "horizontal_flip": {
        "on": True
    },
    "vertical_flip": {
        "on": True
    },
    "gaussian_blur": {
        "on": True,
        "params": {
            "kernel_size": 5,
            "sigma": (0.1, 2.0)
        }
    }
}



class ImageDataset(Dataset):
  def __init__(self, csvfilePath, transformConfig=transformConfig):
    self.csvfilePath = csvfilePath
    self.dataframe = pd.read_csv(csvfilePath)
    self.datasetType = self._checkTrainTest()
    self.config = self._loadConfig()

    self.datasize = self.dataframe.shape[0]
    self.numLabel = self.dataframe['label_id'].nunique()
    self.labeldict = {idnum: self.dataframe.index[self.dataframe['label_id'] == idnum].to_list() for idnum in self.dataframe['label_id'].value_counts().index}
    self.transform = v2.Compose([
      v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
      v2.PILToTensor(),
      v2.ConvertImageDtype(torch.float32)
    ])
    self.transformConfig = transformConfig

    dataLastModified = os.stat(self.csvfilePath).st_mtime
    self.dataLastModified = datetime.fromtimestamp(dataLastModified).strftime('%Y-%m-%d %H:%M:%S')
  
  def __len__(self):
    return self.datasize
  
  def __getitem__(self, idx):
    img = Image.open(self.getImagePath(idx))
    img = self.transform(img) # apply the transform to the image
    transformlog = [""] # use pass by ref to record the transform info

    # apply augmentations
    img = self._random_rotation(img, transformlog)
    img = self._random_horizontal_flip(img, transformlog)
    img = self._random_vertical_flip(img, transformlog)
    img = self._random_gaussian_blur(img, transformlog)

    label = torch.tensor(self.getLabelId(idx), dtype = torch.long) # convert the label to a tensor
    return img, label, transformlog[0]
  
  def getImagePath(self, idx):
    row = self.dataframe.iloc[idx]
    return row['image_path']
  
  def getLabelId(self, idx):
    row = self.dataframe.iloc[idx]
    return row['label_id']
  
  def _checkTrainTest(self):
    if "test_info" in self.csvfilePath:
      return "TestSet"
    
    if "train_info" in self.csvfilePath:
      return "TrainSet"
    
    else:
      return "Others"
  
  def _loadConfig(self):
    with open("../../config.json", "r") as f:
      allConfig = json.load(f)
    
    config = allConfig["dataset"]["ImageDataset"]

    return config
  
  def _random_gaussian_blur(self, img, transformlog):
    """Gaussian blur with 50% probability"""
    if self.transformConfig["gaussian_blur"]["on"]:
      params = self.transformConfig["gaussian_blur"]["params"]
      if random.random() < 0.5:
          transformlog[0] += f"Gaussian Blur: {str(params)}\n"
          return v2.GaussianBlur(**params)(img)

    return img

  def _random_rotation(self, img, transformlog):
    """Random rotation between -45 to 45 degrees with a 50% probability"""
    if self.transformConfig["rotation"]["on"]:
      params = self.transformConfig["rotation"]["params"]
      if random.random() < 0.5:
          angle = random.uniform(*params["degrees"])
          transformlog[0] += f"Rotating by {angle:.2f} degrees\n"
          return v2.RandomRotation(degrees=(angle, angle))(img)
    
    return img

  def _random_horizontal_flip(self, img, transformlog):
    """Horizontal flip with 50% probability"""
    if self.transformConfig["horizontal_flip"]["on"]:
      if random.random() < 0.5:
          transformlog[0] += "Applying Horizontal Flip\n"
          return v2.RandomHorizontalFlip()(img)
      
    return img

  def _random_vertical_flip(self, img, transformlog):
    """Vertical flip with 50% probability"""
    if self.transformConfig["horizontal_flip"]["on"]:
      if random.random() < 0.5:
          transformlog[0] += "Applying Vertical Flip\n"
          return v2.RandomVerticalFlip()(img)
      
    return img

  
  def visualizeAndSave(self, idx, savedPath='../img/'):
    os.makedirs(savedPath, exist_ok=True) # make a dir if not exists
    titleBefT_params = self.config["visualizeAndSave"]["title_before_transform"]
    titleAftT_params = self.config["visualizeAndSave"]["title_after_transform"]
    label_params = self.config["visualizeAndSave"]["label_text_params"]
    label_params['s'] = label_params['s'].format(self.getLabelId(idx))
    label_transformlogs = self.config["visualizeAndSave"]["label_transform_logs"]

    
    timeNow = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d%H%M%S")
    filename = f"{savedPath}{self.datasetType}_{timeNow}_{idx}.png"

    ### Image Before Transform ###
    imgBefT = Image.open(self.getImagePath(idx))
    
    ### Image After Transform ###
    res = self.__getitem__(idx)
    imgAftT = res[0].clone().detach().cpu()
    label_transformlogs['s'] = label_transformlogs['s'].format(res[2])
    imgAftTnp = imgAftT.numpy().transpose(1, 2, 0)
    # img_np = img_np.clip(img_np, 0, 1)


    ### Plot ###
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(imgBefT)
    ax[0].axis('off')
    ax[1].imshow(imgAftTnp)
    ax[1].axis('off')
    fig.text(**titleBefT_params)
    fig.text(**titleAftT_params)
    fig.text(**label_params)
    fig.text(**label_transformlogs)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

  
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
 


if __name__ == "__main__":
  testData = ImageDataset('../data/train_info.csv')
  # print(testData.__getitem__(0)) # displays the first image and label as a tensor
  # # print(testData[0])
  # # testData.summary()
  # print(testData.labeldict[2][:10])
  # print(testData[134][1])
  # # print(len(testData))

  # # TEST CASES
  # # ensure data is a valid dataframe
  # assert isinstance(testData.dataframe, pd.DataFrame)

  # # check column names
  # expected_columns = ['image_path', 'image_id', 'label_id', 'label_text', 'label_raw', 'source']
  # assert testData.dataframe.columns.tolist() == expected_columns

  # # verify shape and non-emptiness
  # assert testData.dataframe.shape[1] == 6
  # assert not testData.dataframe.empty

  # # display DataFrame information
  # print("DatafFrame Info:")
  # print(testData.dataframe.info())
  # print("Random Sample Rows:")
  # print(testData.dataframe.sample(5))

  # # print labels sorted by label_id
  # sorted_labels = testData.dataframe[['label_text', 'label_id']].drop_duplicates().sort_values(by='label_id')
  # for index, row in sorted_labels.iterrows():
  #   print(f"Label: {row['label_text']}, Label ID: {row['label_id']}")

  # # testing __len__()
  # assert len(testData) == testData.dataframe.shape[0]
  # print(f"Dataset Length: {len(testData)}")

  # # testing __getitem__()
  # img, label = testData[0]
  # print(f"Image shape: {img.shape}, Label: {label}")
  # ## check types and properties of returned tensors
  # assert isinstance(img, torch.Tensor)
  # assert isinstance(label, torch.Tensor)
  # assert label.dtype == torch.long
  # ## debug print statements for tensor shape and label
  # print(f"Image Tensor Shape: {img.shape}")
  # print(f"Label Tensor Shape: {label.shape}, Value: {label.item()}")
  # print(f"Row at index 0:\n{testData.dataframe.iloc[0]}")

  # # testing summary()
  # testData.summary()
  # assert testData.datasize > 0
  # assert isinstance(testData.labeldict, dict)
  # assert testData.numLabel == len(testData.labeldict)
  testData.visualizeAndSave(8)
  # print(testData[8])
