import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations
from albumentations.pytorch import ToTensor, ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())
import cv2


"""Loading the dataframe and filenames"""
path         = Path("....Kaggle/Cassava-Leaf-Disease-Classification/Data")
df_path      = path/'train.csv'
train_path   = path/'train_images'
train_fnames = train_path.ls()


df = pd.read_csv(df_path)
num_classes = df['label'].nunique()
df['label'].value_counts()


###Train/Validation Split##
"""I'll split the main dataframe into train_df and val_df with stratification based on the labels. We are using 20% of the data for validation."""
stafy = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
for train_idx, valid_idx in stafy.split(X=df, y=df['label']):
    train_df = df.loc[train_idx]
    valid_df = df.loc[valid_idx]


mean = [0.4589, 0.5314, 0.3236]
std  = [0.2272, 0.2297, 0.2200]


"""Dataset and DataLoaders"""
train_tfms =    albumentations.Compose([
                albumentations.RandomResizedCrop(256,256),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.ShiftScaleRotate(p=0.5),
                albumentations.HueSaturationValue(
                    hue_shift_limit=0.2,
                    sat_shift_limit=0.2,
                    val_shift_limit=0.2,
                    p=0.5
                ),
                albumentations.RandomBrightnessContrast(
                    brightness_limit=(-0.1,0.1),
                    contrast_limit=(-0.1,0.1),
                    p=0.5
                ),
                albumentations.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                    p=1.0
                ),
                albumentations.CoarseDropout(p=0.5),
                albumentations.Cutout(p=0.5),
                ToTensorV2()], p=1.0)


valid_tfms =    albumentations.Compose([
                albumentations.CenterCrop(256,256,p=1.0),
                albumentations.Resize(256,256),
                albumentations.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                    p=1.0
                ),
                ToTensorV2()], p=1.0)


class LeafData(Dataset):
    def __init__(self, df, split='train'):
        if split == 'train':
            self.transforms = train_tfms
        elif split == "valid":
            self.transforms = valid_tfms

        self.paths  = [train_path/id_ for id_ in df['image_id'].values]
        self.labels = df['label'].values

    def __getitem__(self, idx):
        img   = cv2.imread(str(self.paths[idx]))[..., ::-1] # ::-1 is here because cv2 loads the images in BGR rather than RGB
        img   = self.transforms(image=img)['image']
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.paths)


def make_dataloaders(batch_size=8, num_workers=0, pin_memory=True, **kwargs):
    dataset    = LeafData(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                    pin_memory=pin_memory, shuffle=True if kwargs['split'] == 'train' else False) 
    
    return dataloader


train_dl = make_dataloaders(df=train_df, split='train')
valid_dl = make_dataloaders(df=valid_df, split='valid')