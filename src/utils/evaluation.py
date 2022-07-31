# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

class EvalGAN:
    """
    Network Monitoring

    Args:
        dataset: Name of dataset.
        epoch: Start epoch for training.
        n_epoch: End epoch for training.
        lr: learning rate.
        
    Returns:
        Save monitoring figures.
    """
    def __init__(self, dataset, epoch, n_epoch, lr):
        self.dataset = dataset
        self.epoch = epoch
        self.n_epoch = n_epoch
        self.track_loss_g = np.array([])
        self.track_loss_d = np.array([])
        self.track_loss_adv = np.array([])
        self.lr = lr
        self.epochs_g = []
        self.epochs_d = []
        self.epochs_adv = []
    
    def add_batch_loss(self, loss_g, loss_d, loss_adv):
        """Adds mean loss of a batch"""
        self.track_loss_g = np.append(self.track_loss_g, loss_g)
        self.track_loss_d = np.append(self.track_loss_d, loss_d)
        self.track_loss_adv = np.append(self.track_loss_adv, loss_adv)
    
    def add_epoch_loss(self):
        """Adds mean loss of an epoch"""
        epoch_mean_g = np.mean(self.track_loss_g)
        epoch_mean_d = np.mean(self.track_loss_d)
        epoch_mean_adv = np.mean(self.track_loss_adv)
        self.epochs_g.append(epoch_mean_g.item())
        self.epochs_d.append(epoch_mean_d.item())
        self.epochs_adv.append(epoch_mean_adv.item())

        self.track_loss_g = self.track_loss_d = self.track_loss_adv = np.array([])
        return epoch_mean_g.item()

    def write_loss_to_file(self, dest):
        """Write monitoring to file"""
        with open(dest, "w+") as txt_file:
            txt_file.write(str(self.epochs_g) + "\n" + str(self.epochs_d) + "\n" + str(self.epochs_adv) + "\n")
            

"""
PyTorch SSIM computation
https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
"""
def gaussian(window_size, sigma):
    """Compute gaussian blurre"""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    """Compute SSIM of two tensors"""
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    C3 = C2/2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    # ref: https://www.mathworks.com/help/images/ref/ssim.html
    luminance = (2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)
    contrast = (2*sigma1_sq*sigma2_sq + C2)/(sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C3)/(sigma1_sq * sigma2_sq + C2)

    if size_average:
        return ssim_map.mean(), luminance.mean(), contrast.mean(), structure.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1), luminance.mean(1).mean(1).mean(1), contrast.mean(1).mean(1).mean(1), structure.mean(1).mean(1).mean(1)


def tensor_ssim(img1, img2, window_size = 11, size_average = True):
    """Calls SSIM function"""
    return _ssim(img1, img2, window, window_size, channel, size_average)

        

