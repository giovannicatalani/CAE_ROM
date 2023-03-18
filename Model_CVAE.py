# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:09:11 2023

@author: giosp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self,lat_dim, y_dim):
        super(ConditionalVariationalAutoencoder, self).__init__()
        """
          Initialize the VAE module.
          This model takes as input the images and learns their latent
          representations
          
          Parameters:
          latent_dim (int): Size of the latent vector (es. 5,10,20)
        """
        #Encoder
        self.lat_dim = lat_dim
        self.y_dim = y_dim
        
        
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
            nn.Flatten())
        
        self.conditional_encoder = nn.Sequential(
            nn.Linear(in_features=16*14*8 + self.y_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.25))
            
        
        self.fc_mu = nn.Linear(128, lat_dim)
        self.fc_logvar = nn.Linear(128, lat_dim)
        
        self.intermediate = nn.Sequential(
            nn.Linear(in_features=lat_dim + self.y_dim, out_features=128),
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
        
        
    def encode(self, x, y):
        x = self.encoder(x)
        xy = torch.cat([x, y], dim=1)
        xy = self.conditional_encoder(xy)
        mu = self.fc_mu(xy)
        logvar = self.fc_logvar(xy)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        zy = self.intermediate(zy)
        zy = zy.reshape(zy.shape[0], -1, 14, 8)
        zy = self.decoder(zy)
        return zy

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar
    
    def sample(self, num_samples):
        z = torch.randn((num_samples, self.lat_dim)).to(next(self.parameters()).device)
        samples = self.decode(z)
        return samples

    
def vae_loss(recon_x, x, mu, logvar, kl_weight = 1.0):
    #bce_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') # reconstruction loss
    mse_loss = F.mse_loss(recon_x, x, reduction='sum') # additional MSE loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL divergence loss
    loss = mse_loss + kl_weight * kld_loss
    return loss, mse_loss, kld_loss
