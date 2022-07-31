# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

def pixel_threshold(x, threshold):
    """Sets all pixels of image > threshold to RGB value"""
    x = np.squeeze(x, axis=0)
    x = np.transpose(x, (1, 2, 0))
    x[x > threshold] = 0.0

    new_img = np.zeros((x.shape[1], x.shape[2], 3), np.uint8)    
    new_img[x[:, :, :,] >= threshold, 0] = 189
    new_img[x[:,] >= threshold, 1] = 131
    new_img[x[:,] >= threshold, 2] = 59
    return new_img


def colorize_one_hot_tensor(arr, channels):
    """Numpy Array: Assigns RGB values to classes of one-hot-tensor"""
    arr = np.transpose(arr[0], (1, 2, 0))
    img_arr = np.zeros((arr.shape[0], arr.shape[1], channels))

    colors = [(0,0,0),(255, 255, 0), (170, 0, 51), (85, 85, 0), (0, 255, 255), (85, 51, 0),
        (170, 255, 85), (255, 0, 0), (51, 170, 221), (0, 85, 85), (255, 85, 0), (255, 170, 0),
        (128, 0, 0), (0, 119, 221), (85, 255, 170), (0, 0, 255),  (255, 0, 0), (0, 128, 0),
        (52, 86, 128), (0, 0, 85), (0, 85, 0)]

    for i in range(arr.shape[0]):             
        for j in range(arr.shape[1]):
            index = np.argmax(arr[i,j])
            img_arr[i,j] = colors[index.item(0)] 
    
    return img_arr

def indexing_one_hot_tensor(arr, channels):
    """Creates class tensor with indices from best classes of one-hot-tensor"""
    arr = np.transpose(arr[0], (1, 2, 0))
    img_arr = np.zeros((arr.shape[0], arr.shape[1], channels))

    for i in range(arr.shape[0]):             
        for j in range(arr.shape[1]):
            index = np.argmax(arr[i,j])
            img_arr[i,j] = (index.item(0), index.item(0), index.item(0))
    
    return img_arr

def colorize(tensor):
    """Torch Tensor: Assigns RGB values to classes of one-hot-tensor"""
    arr = np.transpose(arr[0], (1, 2, 0))
    colors = [(0,0,0),(255, 255, 0), (170, 0, 51), (85, 85, 0), (0, 255, 255), (85, 51, 0),
    (170, 255, 85), (255, 0, 0), (51, 170, 221), (0, 85, 85), (255, 85, 0), (255, 170, 0),
    (128, 0, 0), (0, 119, 221), (85, 255, 170), (0, 0, 255),  (255, 0, 0), (0, 128, 0),
    (52, 86, 128), (0, 0, 85), (0, 85, 0)]
    
    tensor = torch.argmax(tensor, dim=1)
    nc = 20

    tensor = torch.unsqueeze(tensor, 1)
    tensor = torch.transpose(tensor, 1,3)
    ts = tensor.size()

    r = torch.zeros_like(tensor)
    g = torch.zeros_like(tensor)
    b = torch.zeros_like(tensor)

    for i in range(tensor.size()[0]):
        for c in range(0,nc):
            idx = tensor == c
            r[idx] = colors[c][0]
            g[idx] = colors[c][1]
            b[idx] = colors[c][2]

    rgb = torch.cat([r, g, b], dim=3).cpu()
    rgb = torch.transpose(rgb, 3,1).type(torch.uint8)

    rgb = [transforms.ToPILImage()(m) for m in rgb]

    tensor_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    tensor_transforms = transforms.Compose(tensor_transforms)
    
    for i, img in enumerate(rgb):
        img = tensor_transforms(img)
        img = torch.unsqueeze(img, 0)
        if i == 0:
            new_tensor = img
        else:
            new_tensor = torch.cat((new_tensor, img), 0)
    new_tensor = new_tensor.type(torch.cuda.FloatTensor)

    return new_tensor

def indexing(tensor):
        tensor = torch.argmax(tensor, dim=1)
        tensor = torch.unsqueeze(tensor, 1)
        return tensor.type(torch.cuda.FloatTensor)