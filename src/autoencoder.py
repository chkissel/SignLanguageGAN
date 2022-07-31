# -*- coding: utf-8 -*-
"""Autoencoder

This is a modified and strongly extended version of
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/pix2pix.py.
"""

import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from src.dataloader import * 
from src.utils.models import *
from src.utils.post_processing import *
from src.utils.evaluation import *

import torch.nn as nn
import torch.nn.functional as F
import torch

class AutoEncoder:
    """
    Autoencoder Class

    Args:
        config: A dictionary containing network configurations.
        
    Returns:
        Train: Trains PyTorch model.
    """
    def __init__(self, config, mode):

        self.mode = mode
        self.images_path = config.images_dir
        self.conditions_path = config.conditions_dir
        self.targets_path = config.targets_dir
        self.epoch = config.epoch
        self.n_epochs = config.n_epochs
        self.dataset_name = config.dataset_name
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.b1 = config.b1
        self.b2 = config.b2
        self.n_cpu = config.n_cpu
        self.img_height = config.img_height
        self.img_width = config.img_width
        self.channels = config.channels
        self.loss = config.loss
        self.sample_interval = config.sample_interval
        self.checkpoint_interval = config.checkpoint_interval


    def initialize(self):
        """initialize networks and set-up infrastructure"""
        os.makedirs(f"images/{self.dataset_name}", exist_ok=True)
        os.makedirs(f"saved_models/{self.dataset_name}", exist_ok=True)
        os.makedirs(f"eval/{self.dataset_name}", exist_ok=True)
        
        # Select generator loss function
        if self.loss == 'MSE':
            self.criterion_pixelwise = torch.nn.MSELoss() 
            self.lambda_pixel = 1
        elif self.loss == 'L1':
            self.criterion_pixelwise = torch.nn.L1Loss()
            self.lambda_pixel = 100
        elif self.loss == 'CrossEntropy':
            self.criterion_pixelwise = torch.nn.CrossEntropyLoss()
            self.lambda_pixel = 1

        # Initialize generator
        if self.mode == 'conditional':
            self.generator = GeneratorUNet(in_channels=2, out_channels=1)
        else: 
            self.generator = GeneratorUNet()

        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
            self.criterion_pixelwise.cuda()
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

        if self.epoch != 0:
            # Load pretrained models
            self.generator.load_state_dict(torch.load(f"saved_models/{self.dataset_name}/generator_{self.epoch}.pth"))
        else:
            # Initialize weights
            self.generator.apply(weights_init_normal)

        # Initialize Adam optimizer
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

    def load_data(self):
    """Load batch from dataset"""

        # Configure dataloaders
        transforms_ = [
            transforms.Resize((self.img_height, self.img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]

        data_paths = {
            'imgs_dir': self.images_path,
            'targets_dir': self.targets_path,
        }
            
        if self.mode == 'conditional':
            data_paths['conditions_dir'] = self.conditions_path

        # Basic datalaoder
        self.dataloader = DataLoader(
            DataReader(data_paths, transforms_),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
        )

        # Validation loader
        self.val_dataloader = DataLoader(
            loader(data_paths, transforms_),
            batch_size=3,
            shuffle=True,
            num_workers=1,
        )

    def log(self, epoch, i, loss_G):
        # Print log to terminal
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
            % (
                epoch,
                self.n_epochs,
                i,
                len(self.dataloader),
                loss_G.item(),
            )
        )

    def sample_images(self, batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(self.val_dataloader))
        real_A = Variable(imgs["image"].type(self.Tensor))
        real_B = Variable(imgs["target"].type(self.Tensor))

        if self.mode == 'conditional':
            condition = Variable(imgs["condition"].type(self.Tensor))
            concat = torch.cat((real_A, condition), dim=1)
            fake_B = self.generator(concat)
        else:
            fake_B = self.generator(real_A)
        
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, f"images/{self.dataset_name}/{batches_done}_{epoch}.png", nrow=5, normalize=True)

    def train(self):
        """Train Autoencoder"""
        self.initialize()
        self.load_data()

        for epoch in range(self.epoch, self.n_epochs):
            for i, batch in enumerate(self.dataloader):

                # Model inputs
                real_A = Variable(batch["image"].type(self.Tensor))
                real_B = Variable(batch["target"].type(self.Tensor))

                if self.mode == 'conditional':
                    condition = Variable(batch["condition"].type(self.Tensor))
                    real_A = torch.cat((real_A, condition), dim=1)

                self.optimizer_G.zero_grad()

                fake_B = self.generator(real_A)

                # Pixel-wise loss
                loss_G = self.criterion_pixelwise(fake_B, real_B) * self.lambda_pixel

                loss_G.backward()

                self.optimizer_G.step()

                # Determine approximate time left
                batches_done = epoch * len(self.dataloader) + i
                
                # Log
                self.log(epoch, i, loss_G)

                batch_nr = str(epoch) + '_' + str(i)
                # If at sample interval save image
                if batches_done % self.sample_interval == 0:
                    self.sample_images(batch_nr)

                #self.evaluate.add_loss(loss_G.item())

            if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.generator.state_dict(), f"saved_models/{self.dataset_name}/generator_{epoch}.pth")

            if epoch == self.n_epochs -1:
                # Save model checkpoints
                torch.save(self.generator.state_dict(), f"saved_models/{self.dataset_name}/discriminator_{epoch}.pth")
            
