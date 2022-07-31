# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader

import glob
import random

import cv2
import numpy as np
import json
import csv

from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
from src.utils.padding import padding

import sys


class DataReader(Dataset):
    """
    Basic Data Reader 

    Args:
        params: A dictionary containing the paths to data.
        transforms_: A composition of several torchvision transformations.
        padding: A boolean value if padding should be applied.
        
    Returns:
        A dictionary with data samples.
    """
    def __init__(self, params, transforms_: None, padding=None):
        
        self.transform = transforms.Compose(transforms_)
        self.padding = padding
       
        self.img_dirs = sorted(glob.glob(params['imgs_dir'] + '*'))
        self.target_dirs = sorted(glob.glob(params['targets_dir'] + '*'))
        self.condition_dirs = sorted(glob.glob(params['conditions_dir'] + '*'))
    def __len__(self) -> int:
        """compute the lenght of the dataset"""
        return len(self.img_dirs)

    def __getitem__(self, idx: int) -> dict:
        """return dict with sample"""

        if idx > self.__len__():
            raise IndexError

        img_dir = sorted(glob.glob(self.img_dirs[idx] + '/*.png'))
        condition_dir = sorted(glob.glob(self.condition_dirs[idx] + '/*.jpg'))
        target_dir = sorted(glob.glob(self.target_dirs[idx] + '/*.png'))

        # choose two random indices in the range of the image dir
        indices = list(range(len(img_dir) - 1))
        random.shuffle(indices)
        img_idx = indices.pop()
        tgt_idx = indices.pop()

        image = Image.open(img_dir[img_idx]) 
        condition = Image.open(condition_dir[tgt_idx]) 
        target = Image.open(target_dir[tgt_idx]) 

        if self.padding != None:
            image = padding(image, self.padding)
            condition = padding(condition, self.padding)
            target = padding(target, self.padding)

        image = self.transform(image)
        condition = self.transform(condition)
        target = self.transform(target)

        # create return dict
        sample = {"image": image, "condition": condition, "target": target}

        return sample


class EasyReader(Dataset):
    """
    Custom Data Reader

    Args:
        folder: Path to image folder.
        transforms_: A composition of several torchvision transformations.
        
    Returns:
        A dictionary with data samples.
    """
    def __init__(self, folder, transforms_: None):
        
        self.transform = transforms.Compose(transforms_)
        self.img_dir = sorted(glob.glob(folder + '/*.png'))

    def __len__(self) -> int:
        """compute the lenght of the dataset"""
        return len(self.img_dir)

    def __getitem__(self, idx: int) -> dict:
        """return dict with sample"""
        
        if idx > self.__len__():
            raise IndexError

        image = Image.open(self.img_dir[idx]) 
        
        image = self.transform(image)
        
        # create return dict
        sample = {"image": image}

        return sample


class MSASLReader(Dataset):
    def __init__(self, params, transforms_: None, padding=None):
        """
        Custom Data Reader for MSASL Dataset

        Args:
            params: A dictionary containing the paths to data.
            transforms_: A composition of several torchvision transformations.
            padding: A boolean value if padding should be applied.
            
        Returns:
            A dictionary with data samples.
        """
        
        self.transform = transforms.Compose(transforms_)
        self.padding = padding
       
        self.img_dirs = sorted(glob.glob(params['imgs_dir'] + '*'))
        self.target_dirs = sorted(glob.glob(params['targets_dir'] + '*'))
        
        # set seed for reproducibility
        #random.seed(0)

    def __len__(self) -> int:
        """compute the lenght of the dataset"""
        return len(self.img_dirs) -1

    def __getitem__(self, idx: int) -> dict:
        """return dict with sample"""

        if idx > self.__len__():
            raise IndexError
        
        self.idx = idx
        img_dir = sorted(glob.glob(self.img_dirs[idx] + '/*.jpg'))
        condition_dir = sorted(glob.glob(self.img_dirs[idx] + '/poses/*.png'))
        target_dir = sorted(glob.glob(self.target_dirs[idx] + '/*.jpg'))
        
        # load MSE differences from JSON file
        with open(self.img_dirs[idx] + '/mse.json') as json_file:
            self.diff = [float(item) for item in json.load(json_file)[1:-1].split(',')]
            self.diff = np.asarray(self.diff)
            # insert 0.0 at the beginning of array for first frame
            self.diff = np.insert(self.diff, 0, 0.0, axis=0)
        
        # run  find_target() to apply  sampeling  strategy
        img_idx, tgt_idx = self.find_indices() 

        image = Image.open(img_dir[img_idx]) 
        condition = Image.open(condition_dir[tgt_idx]) 
        target = Image.open(target_dir[tgt_idx]) 

        if self.padding != None:
            image = padding(image, self.padding)
            condition = padding(condition, self.padding)
            target = padding(target, self.padding)

        image = self.transform(image)
        condition = self.transform(condition)
        target = self.transform(target)

        # create return dict
        #sample = {"image": image, "condition": condition, "target": target, "gloss": self.img_dirs[idx]}
        sample = {"image": image, "condition": condition, "target": target}

        return sample

    def find_indices(self):
        """returns indices for input and target frame"""
        # sort diff values ascending and randomly pick image index from lowest quarter
        low_indices = np.argpartition(self.diff, int((len(self.diff) - 1) / 4))
        img_idx = random.choice(low_indices[:int((len(self.diff) - 1) / 4)])
        
        value = self.diff[img_idx]
        
        # differences to image value
        dists_to_value = np.abs(self.diff - value)
        max_diff = dists_to_value.max()
        min_diff = dists_to_value.min()

        # all values that are further away from image than half the maximum difference
        distant_values = dists_to_value[dists_to_value > (max_diff - min_diff) / 2]

        # pick random value from distant_values and find index in dists_to_value
        try:
            # pick random value from distant_values and find index in dists_to_value
            dist = distant_values[random.randint(0, len(distant_values) - 1)] 
            return img_idx, dists_to_value.tolist().index(dist)
        except:
            # if mse didn't work properly pick random index
            return img_idx, random.randint(0, len(self.diff) - 1)


class GlReader(Dataset):
    """
    Custom Data Reader for Gebaerdenlernen  Dataset

    Args:
        root_dir: A dictionary containing the annotation file.
        transforms_: A composition of several torchvision transformations.        
    Returns:
        A dictionary with data samples.
    """
    def __init__(self, root_dir: str, transforms_=None) -> str:
        
        self.transform = transforms.Compose(transforms_)
        self.data_dir = root_dir 

        self.annotation = []

        with open( self.data_dir + '/annotations/gebaerdenlernen.mp4.csv', encoding='utf_8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
            for row in csv_reader:
                self.annotation.append(row)
        self.annotation = self.annotation[1:]

    def __len__(self) -> int:
        """compute the lenght of the dataset"""
        return len(self.annotation)

    def __getitem__(self, idx: int) -> dict:
        """return dict with sample"""
        if idx > self.__len__():
            raise IndexError

        gloss = self.annotation[idx][0]
        video_url = self.annotation[idx][2]

        images_dir = self.data_dir + '/features/frames/' + video_url[13:-4] + '/*.jpg'
        conditions_dir = self.data_dir + '/features/images/' + video_url[13:-4] + '/*.jpg'

        images_dir = sorted(glob.glob(images_dir))
        conditions_dir = sorted(glob.glob(conditions_dir))

        # find random indices for input and target
        try:
            img_idx = random.randint(0, len(images_dir) - 1)
            tgt_idx = random.randint(0, len(images_dir) - 1)
        except:
            img_idx = 0
            tgt_idx = 3

        image = Image.open(images_dir[img_idx]) 
        condition = Image.open(conditions_dir[tgt_idx]) 
        target = Image.open(images_dir[tgt_idx]) 

        image = self.transform(image)
        condition = self.transform(condition)
        target = self.transform(target)
        # create return dict
        sample = {"image": image, "condition": condition, "target": target}

        return sample


class RandomGlReader(Dataset):
    """
    Custom Data Reader for Gebaerdenlernen 
    with random glosse selection throughout all videos

    Args:
        root_dir: A dictionary containing the annotation file.
        transforms_: A composition of several torchvision transformations.        
    Returns:
        A dictionary with data samples.
    """
    def __init__(self, root_dir: str, transforms_=None) -> str:
        
        self.transform = transforms.Compose(transforms_)
        self.data_dir = root_dir 

        self.annotation = []

        with open( self.data_dir + '/annotations/gebaerdenlernen.mp4.csv', encoding='utf_8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
            for row in csv_reader:
                self.annotation.append(row)
        self.annotation = self.annotation[1:]

    def __len__(self) -> int:
        """compute the lenght of the dataset"""
        return len(self.annotation)

    def __getitem__(self, idx: int) -> dict:
        """return dict with sample"""
        if idx > self.__len__():
            raise IndexError

        gloss = self.annotation[idx][0]
        video_url = self.annotation[idx][2]

        random_idx = random.randint(0, len(self.annotation) - 1)
        random_video = self.annotation[random_idx][2]

        images_dir = self.data_dir + '/features/frames/' + video_url[13:-4] + '/*.jpg'
        targets_dir = self.data_dir + '/features/frames/' + random_video[13:-4] + '/*.jpg'
        conditions_dir = self.data_dir + '/features/images/' + random_video[13:-4] + '/*.jpg'
        images_dir = sorted(glob.glob(images_dir))
        targets_dir = sorted(glob.glob(targets_dir))
        conditions_dir = sorted(glob.glob(conditions_dir))

        # find random indices for input and target
        try:
            img_idx = random.randint(0, len(images_dir) - 1)
            tgt_idx = random.randint(0, len(targets_dir) - 1)
        except:
            img_idx = 0
            tgt_idx = 3

        image = Image.open(images_dir[img_idx]) 
        condition = Image.open(conditions_dir[tgt_idx]) 
        target = Image.open(targets_dir[tgt_idx]) 

        image = self.transform(image)
        condition = self.transform(condition)
        target = self.transform(target)
        # create return dict
        sample = {"image": image, "condition": condition, "target": target}

        return sample

    def pad_collate(batch: list) -> dict:
        """customized pad collate method"""
        batch_size = len(batch)
        # if arrays are not torch tensors make them
        if not isinstance(batch[0]['image'], torch.Tensor):
            for sample in batch:
                image = [img.transpose((2, 0, 1)) for img in sample['image']]
                condition = [pose.transpose((2, 0, 1)) for pose in sample['condition']]
                target = [trgt.transpose((2, 0, 1)) for trgt in sample['target']]
                sample['image'] = torch.tensor(np.array(video), dtype=torch.float32)
                sample['condition'] = torch.tensor(np.array(pose), dtype=torch.float32)
                sample['target'] = torch.tensor(np.array(pose), dtype=torch.float32)

        # retrieve sample meta information
        _, C, H, W = batch[0]['image'].size()
        image_count = [sample['image'].size(0) for sample in batch]
        padded_images = torch.zeros((batch_size, max(frame_count), C, H, W))

        _, C, H, W = batch[0]['condition'].size()
        pose_count = [sample['condition'].size(0) for sample in batch]
        padded_poses = torch.zeros((batch_size, max(pose_count), C, H, W))

        _, C, H, W = batch[0]['target'].size()
        target_count = [sample['target'].size(0) for sample in batch]
        padded_targets = torch.zeros((batch_size, max(pose_count), C, H, W))
        # fill up frames with zeros.
        for sample_idx in range(batch_size):
            padded_images[sample_idx][:image_count[sample_idx]] = batch[sample_idx]['image']
            padded_poses[sample_idx][:poses_count[sample_idx]] = batch[sample_idx]['condition']
            padded_targets[sample_idx][:target_count[sample_idx]] = batch[sample_idx]['target']

        return {'image': padded_images, 'condition': padded_poses, 'target': padded_targets}

class RWTHReader(Dataset):
    def __init__(self, params, transforms_: None, padding=None):
        """
        Custom Data Reader for RWTH-Phoenix-14T Dataset

        Args:
            params: A dictionary containing the paths to data.
            transforms_: A composition of several torchvision transformations.
            padding: A boolean value if padding should be applied.
            
        Returns:
            A dictionary with data samples.
        """
        
        self.transform = transforms.Compose(transforms_)
        self.padding = padding
       
        self.img_dirs = sorted(glob.glob(params['imgs_dir'] + '*'))
        self.target_dirs = sorted(glob.glob(params['targets_dir'] + '*'))
        
        # set seed for reproducibility
        random.seed(0)

    def __len__(self) -> int:
        """compute the lenght of the dataset"""
        return len(self.img_dirs) -1

    def __getitem__(self, idx: int) -> dict:
        """return dict with sample"""

        if idx > self.__len__():
            raise IndexError
        
        self.idx = idx
        img_dir = sorted(glob.glob(self.img_dirs[idx] + '/1/*.png'))
        condition_dir = sorted(glob.glob(self.img_dirs[idx] + '/poses/*.png'))
        target_dir = sorted(glob.glob(self.target_dirs[idx] + '/1/*.png'))
        
       # choose two random indices in the range of the image dir
        indices = list(range(len(img_dir) - 1))
        random.shuffle(indices)
        img_idx = indices.pop()
        tgt_idx = indices.pop()

        image = Image.open(img_dir[img_idx]) 
        condition = Image.open(condition_dir[tgt_idx]) 
        target = Image.open(target_dir[tgt_idx]) 

        if self.padding != None:
            image = padding(image, self.padding)
            condition = padding(condition, self.padding)
            target = padding(target, self.padding)

        image = self.transform(image)
        condition = self.transform(condition)
        target = self.transform(target)

        # create return dict
        #sample = {"image": image, "condition": condition, "target": target, "gloss": self.img_dirs[idx]}
        sample = {"image": image, "condition": condition, "target": target}

        return sample