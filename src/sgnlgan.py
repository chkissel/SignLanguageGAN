# -*- coding: utf-8 -*-
"""SignLanguageGAN

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

import timeit

# Set seed for reproducibility
#torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


class SgnlGAN:
    """
    SignLanguageGAN Pipeline

    Args:
        config: A dictionary containing network configurations.
        
    Returns:
        Train: Trains PyTorch model.
        Test: Tests model on dataset.
    """
    def __init__(self, config):
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

        # Least squares discriminator loss
        self.criterion_GAN = torch.nn.MSELoss() 
        # Calculate output of image discriminator (PatchGAN)
        self.patch = (1, self.img_height // 2 ** 4, self.img_width // 2 ** 4)

        # Initialize generator ans discriminator
        self.generator = DoubleUNet()
        self.discriminator = Discriminator()

        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
            self.criterion_pixelwise.cuda()

            self.discriminator = self.discriminator.cuda()
            self.criterion_GAN.cuda()

            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

        if self.epoch != 0:
            # Load pretrained models
            self.generator.load_state_dict(torch.load(f"saved_models/{self.dataset_name}/generator_{self.epoch}.pth"))
            self.discriminator.load_state_dict(torch.load(f"saved_models/{self.dataset_name}/discriminator_{self.epoch}.pth"))
            # Load benchmark model
            #self.generator.load_state_dict(torch.load(f"saved_models/{self.dataset_name}/generator.pth"))
            #self.discriminator.load_state_dict(torch.load(f"saved_models/{self.dataset_name}/discriminator.pth"))
        else:
            # Initialize weights
            self.generator.apply(weights_init_normal)
            self.discriminator.apply(weights_init_normal)
              
        # Initialize Adam optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Set up monitoring
        self.evaluate = EvalGAN(self.dataset_name, self.epoch, self.n_epochs, self.lr)

        # Initialize human semantic parser
        self.parser_generator = GeneratorUNet(out_channels=20)
        self.predict_generator = GeneratorUNet(in_channels=4)

        if torch.cuda.is_available():
            self.parser_generator = self.parser_generator.cuda()
            self.predict_generator = self.predict_generator.cuda()
        
        # Load pretrained models of human semantic parser stages
        self.parser_generator.load_state_dict(torch.load("saved_models/human_semantic_parser/generator_24.pth"))
        self.predict_generator.load_state_dict(torch.load("saved_models/predict_parser/generator_34.pth"))
        
    def update_lambda(self, i):
        """Compute dynamic lambda"""
        min_lambda = 50
        dynamic_lambda = 150
        abs_cos = (math.cos(i/self.q_lambda) + 1) / 2
        return min_lambda + dynamic_lambda * abs_cos

    def load_data(self):
        """Load batches from dataset"""
        # Configure dataloaders
        self.transforms_ = [
            transforms.Resize((self.img_height, self.img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        # Gebärdenlernen loader
        """
        self.dataloader = DataLoader(
            RandomGlReader('./data', 
            self.transforms_),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
        )
        """
        # MS-ASL loader
        self.dataloader = DataLoader(
            MSASLReader({'imgs_dir': self.images_path,
            'targets_dir': self.targets_path,
            'conditions_dir': self.conditions_path}, 
            self.transforms_),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
        )

    def sample_images(self, batches_done, epoch, real_A, real_B, fake_B):
        """Save image samples
        - light image sampler to save GPU storage -
        """
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, f"images/{self.dataset_name}/{batches_done}_{epoch}.png", nrow=5, normalize=True)  
        img_sample = None

    def parse_and_predict(self, real_A, target_pose):
        """Apply human semantic parser"""
        parsing = self.parser_generator(real_A)
        parsing = indexing(parsing)

        real_A = torch.cat((parsing, target_pose), dim=1) 
        target_pose = None

        # Predict parsing
        fake_B = self.predict_generator(real_A)    
        real_A = None

        # Assign RGB-value to class
        parsing = colorize(parsing)

        return parsing, fake_B

    def train(self):
        """Train SignLanguageGAN"""
        self.initialize()
        self.load_data()
        # Set step size for dynamic lambda
        self.q_lambda = len(self.dataloader)/3
        for epoch in range(self.epoch, self.n_epochs):
            for i, batch in enumerate(self.dataloader):
                try:
                    # Model inputs
                    real_A = Variable(batch["image"].type(self.Tensor))
                    real_B = Variable(batch["target"].type(self.Tensor))
                    condition = Variable(batch["condition"].type(self.Tensor))
                    
                    # Adversarial ground truths
                    valid = Variable(self.Tensor(np.ones((real_A.size(0), * self.patch))), requires_grad=False)
                    fake = Variable(self.Tensor(np.zeros((real_A.size(0), * self.patch))), requires_grad=False)

                    # Apply human semantic parser
                    parsing, pred_parsing = self.parse_and_predict(real_A, condition)
                    
                    encoder_one = torch.cat((condition, pred_parsing.detach()), dim=1)
                    encoder_two = torch.cat((real_A, parsing), dim=1)

                    # ---------------------
                    #  Train generator
                    # ---------------------
                    self.optimizer_G.zero_grad()
                    
                    fake_B = self.generator(encoder_one, encoder_two)
                    
                    pred_fake = self.discriminator(fake_B, condition, pred_parsing, real_A, parsing)
                    loss_GAN = self.criterion_GAN(pred_fake, valid)

                    # Pixel-wise loss
                    loss_pixel = self.criterion_pixelwise(fake_B, real_B)

                    # Compute ssim of generated sample and ground truth
                    #print(tensor_ssim(fake_B, real_B).item())

                    # Total loss
                    loss_G = loss_GAN + self.lambda_pixel * loss_pixel

                    loss_G.backward(retain_graph=True)

                    self.optimizer_G.step()
                    # ---------------------
                    #  Train discriminator
                    # ---------------------
                    
                    self.optimizer_D.zero_grad()

                    # Real loss
                    pred_real = self.discriminator(real_B, condition, pred_parsing, real_A, parsing)
                    loss_real = self.criterion_GAN(pred_real, valid)

                    # Fake loss
                    pred_fake = self.discriminator(fake_B.detach(), condition, pred_parsing, real_A, parsing)
                    loss_fake = self.criterion_GAN(pred_fake, fake)

                    # Total loss
                    loss_D = 0.5 * (loss_real + loss_fake)

                    loss_D.backward()
                    self.optimizer_D.step()
                    
                    
                    batches_done = epoch * len(self.dataloader) + i
                    #batches_left = self.n_epochs * len(self.dataloader) - batches_done
                    
                    # Log to terminal
                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [adv loss: %f]"
                        % (
                            epoch,
                            self.n_epochs,
                            i,
                            len(self.dataloader),
                            loss_D.item(),
                            loss_G.item(),
                            loss_GAN.item(),
                        )
                    )
                    
                    # If at sample interval save image
                    if batches_done % self.sample_interval == 0:
                        self.sample_images(batches_done, epoch, real_A.detach(), real_B.detach(), fake_B.detach())
                        
                    self.evaluate.add_batch_loss(loss_G.item(), loss_D.item(), loss_GAN.item())
                    
                except Exception as e:     
                    with open("log.txt", "w") as log:
                        log.write("Batch failed: {0}\n".format(str(e)))
                    continue
                
                if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
                    # Save model checkpoints
                    torch.save(self.generator.state_dict(), f"saved_models/{self.dataset_name}/intermediate_generator.pth")
                    torch.save(self.discriminator.state_dict(), f"saved_models/{self.dataset_name}/intermediate_discriminator.pth")
                
                if epoch == self.n_epochs -1:
                    # Save model checkpoints
                    torch.save(self.generator.state_dict(), f"saved_models/{self.dataset_name}/generator_{epoch}.pth")
                    torch.save(self.discriminator.state_dict(), f"saved_models/{self.dataset_name}/discriminator_{epoch}.pth")
                
                # Update dynamic lambda
                self.lambda_pixel = self.update_lambda(batches_done)
                
            generator_epoch = self.evaluate.add_epoch_loss()

            self.evaluate.write_loss_to_file(f"eval/{self.dataset_name}/{epoch}_{self.n_epochs}.txt") 
        self.evaluate.write_loss_to_file(f"eval/{self.dataset_name}/{epoch}_{self.n_epochs}.txt")
    
    def test(self):
        """Test SignLanguageGAN"""
        self.initialize()
        self.load_data()
        ssim_values = []
        for epoch in range(self.epoch, self.n_epochs):
            for i, batch in enumerate(self.dataloader):
                # Model inputs
                real_A = Variable(batch["image"].type(self.Tensor))
                real_B = Variable(batch["target"].type(self.Tensor))
                condition = Variable(batch["condition"].type(self.Tensor))

                parsing, pred_parsing = self.parse_and_predict(real_A, condition)
                
                encoder_one = torch.cat((condition, pred_parsing.detach()), dim=1)
                encoder_two = torch.cat((real_A, parsing), dim=1)
                
                fake_B = self.generator(encoder_one, encoder_two)

                ssim = tensor_ssim(fake_B, real_B).item()
                ssim_values.append(ssim)
        ssim_values = np.array(ssim_values)
        print(f"Mean SSIM for test run: {np.mean(ssim_values)}")                      