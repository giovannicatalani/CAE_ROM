# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:13:16 2022

@author: giosp
"""

import torch
import torch.nn as nn

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self,lat_dim):
        super(ConvolutionalAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # Add your encoder layers here
            #First Conv Block
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), #out [64,113,65]
            nn.LeakyReLU(negative_slope=0.25),
            nn.MaxPool2d(kernel_size=2, stride=2), #out [64,56,32]
            #Second Conv Block
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), #out [32,56,32]
            nn.LeakyReLU(negative_slope=0.25),
            nn.MaxPool2d(kernel_size=2, stride=2), #out [32,28,16]
            #Third Conv Block
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), #out [16,28,16]
            nn.LeakyReLU(negative_slope=0.25),
            nn.MaxPool2d(kernel_size=2, stride=2), #out [16,14,8]
            #Flatten and Dense Layer to latent space
            nn.Flatten(),
            nn.Linear(in_features=16*14*8, out_features=128),
            nn.LeakyReLU(negative_slope=0.25),
            nn.Linear(in_features=128, out_features=lat_dim),
            nn.LeakyReLU(negative_slope=0.25))
        
        self.intermediate = nn.Sequential(
            nn.Linear(in_features=lat_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.25),
            nn.Linear(in_features=128, out_features=16*14*8),
            nn.LeakyReLU(negative_slope=0.25))           

    
        self.decoder = nn.Sequential(
            # Add your decoder layers here
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.25),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.25),
            # Upsample the feature maps and pass them through a convolutional layer
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.25),
        
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=1, padding=1, output_padding=0),
            
        )
        
            

    def forward(self, x):
        x_lat = self.encoder(x)
        x = self.intermediate(x_lat)
        x = x.reshape(x.shape[0], -1, 14, 8)
        x = self.decoder(x)
        return x, x_lat