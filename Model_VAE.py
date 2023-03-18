# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:23:30 2023

@author: giosp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    def __init__(self,lat_dim):
        super(VariationalAutoencoder, self).__init__()
        """
          Initialize the VAE module.
          This model takes as input the images and learns their latent
          representations
          
          Parameters:
          latent_dim (int): Size of the latent vector (es. 5,10,20)
        """
        #Encoder
        self.lat_dim = lat_dim
        
        
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
            nn.LeakyReLU(negative_slope=0.25))
        
        self.fc_mu = nn.Linear(128, lat_dim)
        self.fc_logvar = nn.Linear(128, lat_dim)
        
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
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x = self.intermediate(z)
        x = x.reshape(x.shape[0], -1, 14, 8)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def sample(self, num_samples):
        z = torch.randn((num_samples, self.lat_dim)).to(next(self.parameters()).device)
        samples = self.decode(z)
        return samples

    
def vae_loss(recon_x, x, mu, logvar):
    #bce_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') # reconstruction loss
    mse_loss = F.mse_loss(recon_x, x, reduction='sum') # additional MSE loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL divergence loss
    return mse_loss + kld_loss
