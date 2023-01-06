# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 21:29:54 2022

@author: giosp
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from Model import ConvolutionalAutoencoder
import numpy as np

data_directory = 'D:/Thesis/POD_LSTM/data/'
test_directory = data_directory + 'simulations/pitch_a5_A5_f05/'

# Define the device to use for training (e.g. GPU, CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the latent dimension for the autoencoder
latent_dim = 5

# Create the model
model = ConvolutionalAutoencoder(latent_dim)

# Move the model to the device
model = model.to(device)

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
batch_size = 32

train = np.load(data_directory + 'simulations/training_maneuver/labels_flow.npy')
valid = np.load(test_directory + 'labels_flow.npy')
test  = np.load(test_directory + 'labels_flow.npy')
#Dataset
train = torch.from_numpy(train.astype(np.float))
valid = torch.from_numpy(valid.astype(np.float))
test = torch.from_numpy(test.astype(np.float))


p_mean = train.mean((0))#for each channel
p_std = train.std((0))    #for each channel


train = (train-p_mean)/p_std #in pressure
valid = (valid-p_mean)/p_std #in pressure
test  = (test-p_mean)/p_std #in pressure

train = train.unsqueeze(1)

train2 = torch.utils.data.TensorDataset(train,train)
train_loader = torch.utils.data.DataLoader(train2, batch_size=batch_size, num_workers=1, shuffle=True)

valid2 = torch.utils.data.TensorDataset(valid,valid)
valid_loader = torch.utils.data.DataLoader(valid2, batch_size=batch_size, num_workers=1, shuffle=True)

test2 = torch.utils.data.TensorDataset(test,test)
test_loader = torch.utils.data.DataLoader(test2, batch_size=batch_size, num_workers=1, shuffle=True)


print('Dataset creation: done \n')
#%%



# Train the model
for epoch in range(1000):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (inputs, _ ) in enumerate(train_loader):
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        inputs   = inputs.float()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
      

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        
       
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 1000))


print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'model.pt')