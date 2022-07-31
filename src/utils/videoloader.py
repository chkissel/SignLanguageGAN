# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader

import glob

from PIL import Image
import torchvision.transforms as transforms


class VideoReaderMSASL(Dataset):
    """
        Video Reader for MS-ASL Dataset

        Args:
            params: A dictionary containing the paths to data.
            transforms_: A composition of several torchvision transformations.
            padding: A boolean value if padding should be applied.
            
        Returns:
            A dictionary with data samples.
        """
    def __init__(self, root_dir: str, transforms_=None) -> str:
        
        self.transform = transforms.Compose(transforms_)
        self.images_dir = sorted(glob.glob(root_dir + '*.jpg'))
        self.conditions_dir = sorted(glob.glob(root_dir + 'poses/*.png'))

    def __len__(self) -> int:
        """compute the lenght of the dataset"""
        return len(self.images_dir)

    def __getitem__(self, idx: int) -> dict:
        """return dict with sample"""
        if idx > self.__len__():
            raise IndexError

        image = Image.open(self.images_dir[0]) 
        #image = Image.open('test.jpg') 
        condition = Image.open(self.conditions_dir[idx]) 

        image = self.transform(image)
        condition = self.transform(condition)

        # create return dict
        sample = {"image": image, "condition": condition}

        return sample


class VideoReaderGL(Dataset):
    """
        Video Reader for Gebärdenlernen Dataset

        Args:
            params: A dictionary containing the paths to data.
            transforms_: A composition of several torchvision transformations.
            padding: A boolean value if padding should be applied.
            
        Returns:
            A dictionary with data samples.
        """
    def __init__(self, root_dir: str, cond_dir: str, transforms_=None) -> str:
        
        self.transform = transforms.Compose(transforms_)
        self.images_dir = sorted(glob.glob(root_dir + '*.jpg'))
        self.conditions_dir = sorted(glob.glob(cond_dir + '*.jpg'))
    def __len__(self) -> int:
        """compute the lenght of the dataset"""
        return len(self.images_dir)

    def __getitem__(self, idx: int) -> dict:
        """return dict with sample"""
        if idx > self.__len__():
            raise IndexError

        image = Image.open(self.images_dir[0]) 
        condition = Image.open(self.conditions_dir[idx]) 

        image = self.transform(image)
        condition = self.transform(condition)

        # create return dict
        sample = {"image": image, "condition": condition}

        return sample





