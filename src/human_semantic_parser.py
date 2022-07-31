# -*- coding: utf-8 -*-
"""Human Semantic Parser

This is a modified version of
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/pix2pix.py.
"""
import os
import numpy as np
import math
import sys

from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from src.dataloader import * 
from src.utils.lip_loader import *
from src.utils.models import *
from src.utils.evaluation import *
from src.utils.post_processing import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# Set seed for reproducibility
#torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


class HumanSemanticParser:
    """
    Human Semantic Parser Class

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
        
        # Select loss function
        if self.loss == 'L1':
            self.criterion_pixelwise = torch.nn.L1Loss()
            self.lambda_pixel = 100
        elif self.loss == 'CrossEntropy':
            # Weightening background color (black) less strong
            #weight = torch.tensor([0.9, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.2,1,1,1,1,])
            #self.criterion_pixelwise = nn.CrossEntropyLoss(weight=weight)

            self.criterion_pixelwise = nn.CrossEntropyLoss()
            self.lambda_pixel = 1

        # Initialize generator 
        self.generator = GeneratorUNet(out_channels=20)

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

        # Initialize Adam optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Set up monitoring
        self.evaluate = EvalGAN(self.dataset_name, self.epoch, self.n_epochs, self.lr)

    def load_data(self):
        """Load batches from dataset"""
        # Configure dataloaders
        transforms_ = [
            transforms.Resize((self.img_height, self.img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        data_paths = {
            'imgs_dir': self.images_path,
            'targets_dir': self.targets_path,
        }
        
        # LIP loader
        self.dataloader = DataLoader(
            LIPDataset(data_paths, transforms_),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
        )

        # Validation loader
        self.val_dataloader = DataLoader(
            LIPDataset(data_paths, transforms_),
            batch_size=3,
            shuffle=True,
            num_workers=1,
        )

    def log(self, epoch, i, loss_G):
        """Log to terminal"""
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

    def sample_images(self, batches_done, epoch):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(self.val_dataloader))
        real_A = Variable(imgs["image"].type(self.Tensor))
        real_B = Variable(imgs["target"].type(self.Tensor))
        
        fake_B = self.generator(real_A)

        for i, tensor in enumerate(fake_B.split(1)):
            im = colorize_one_hot_tensor(tensor.cpu().detach().numpy(), 3)
            im = im.astype(np.uint8)
            im = Image.fromarray(im)
            im.save("images/%s/%s_%s_fake_%d.png" % (self.dataset_name, batches_done, epoch, i))
             
        save_image(real_A, "images/%s/%s_%s_real_%d.png" % (self.dataset_name, batches_done, epoch, i), normalize=True)

    def train(self):
        """Train Human Semantic Parser"""
        self.initialize()
        self.load_data()
        for epoch in range(self.epoch, self.n_epochs):
            for i, batch in enumerate(self.dataloader):

                # Model inputs
                real_A = Variable(batch["image"].type(self.Tensor))
                real_B = Variable(batch["target"].type(self.Tensor))

                # Discard redundant dimension
                real_B = real_B.squeeze(1)

                # translate to range of 8bit color values
                real_B = real_B * 255
                real_B = real_B.long()
                
                self.optimizer_G.zero_grad()
                fake_B = self.generator(real_A)
               
                loss_G = self.criterion_pixelwise(fake_B, real_B) * self.lambda_pixel
                loss_G.backward()
                self.optimizer_G.step()
                
                batches_done = epoch * len(self.dataloader) + i

                self.log(epoch, i, loss_G)
                
                # If at sample interval save image
                if batches_done % self.sample_interval == 0:
                    self.sample_images(batches_done, epoch)

            if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.generator.state_dict(), f"saved_models/{self.dataset_name}/generator_{epoch}.pth")
            if epoch == self.n_epochs -1:
                # Save model checkpoints
                torch.save(self.generator.state_dict(), f"saved_models/{self.dataset_name}/generator_{epoch}.pth")
            
        