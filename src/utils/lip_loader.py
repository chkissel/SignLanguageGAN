# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import glob

import cv2
import sys 

from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
from src.utils.padding import padding


class LIPDataset(Dataset):
    """
        Custom Data Reader for LIP Dataset

        Args:
            params: A dictionary containing the paths to data.
            transforms_: A composition of several torchvision transformations.
            padding: A boolean value if padding should be applied.
            
        Returns:
            A dictionary with data samples.
        """
    def __init__(self, params, transforms_=None) -> str:
        self.transform = transforms.Compose(transforms_)
        self.imgs_dir = sorted(glob.glob(params['imgs_dir'] + '*.jpg'))
        self.res_dir = sorted(glob.glob(params['targets_dir'] + '*.png'))

    def __len__(self) -> int:
        """compute the lenght of the dataset"""
        return len(self.imgs_dir)

    def __getitem__(self, idx: int) -> dict:
        """return dict with sample"""
        if idx > self.__len__():
            raise IndexError

        image = self.imgs_dir[idx]
        result = self.res_dir[idx]

        image = Image.open(image) 
        result = Image.open(result) 

        #image = padding(image, 256, "RGB")
        #result = padding(result, 256, "P")

        resize = transforms.Resize((256, 256))
        tensorize = transforms.ToTensor()
        
        image = image.convert('RGB')
        #result = result.convert('P')

        result = resize(result)
        result = tensorize(result)

        image = self.transform(image)

        # create return dict
        sample = {"image": image, "target": result}

        return sample






