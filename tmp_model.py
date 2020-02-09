#!/usr/bin/env python3

"""
Defines ML architectures for facade segmentation
"""

import torch
import torch.nn as nn



class cnn(nn.Module):
    """ Contracting-expanding net without skip connections."""

    def __init__(self):
        """ Defines layers, hardcoded size at the moment."""

        super(cnn, self).__init__()

        self.in_size = 256
        self.out_size = 256

        self.activation = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(3, 8, kernel_size = 3, padding = 1)
        self.downconv1 = nn.Conv2d(8, 16, kernel_size = 4, stride = 2, padding = 1)

        self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
        self.downconv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2, padding = 1)

        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        self.downconv3 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)

        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.downconv4 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)

        self.conv5 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)

        self.conv6 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.upconv6 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)

        self.conv7 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        self.upconv7 = nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1)

        self.conv8 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
        self.upconv8 = nn.ConvTranspose2d(32, 16, kernel_size = 4, stride = 2, padding = 1)

        self.conv9 = nn.Conv2d(8, 2, kernel_size = 3, padding = 1)
        self.upconv9 = nn.ConvTranspose2d(16, 8, kernel_size = 4, stride = 2, padding = 1)

    #    self.conv10 = nn.Conv2d(1, 1, kernel_size = 3, padding = 1)
#        self.upconv10 = nn.ConvTranspose2d(8, 1, kernel_size = 4, stride = 2, padding = 1)


        self.model = nn.Sequential(
                    self.conv1, self.activation, self.downconv1,
                    self.conv2, self.activation, self.downconv2,
                    self.conv3, self.activation, self.downconv3,
                    self.conv4, self.activation, self.downconv4,
                    self.conv5, self.activation,
                    self.upconv6, self.conv6, self.activation,
                    self.upconv7, self.conv7, self.activation,
                    self.upconv8, self.conv8, self.activation,
                    self.upconv9, self.conv9#
                    #self.upconv10, self.conv10
                    )

    def forward(self, x):

        out=self.model(x)
        return out
