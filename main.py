# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:27:24 2023

@author: giosp
"""


import torch
import torch.nn as nn
import torch.optim as optim
from Model import ConvolutionalAutoencoder, MLP
import numpy as np
import matplotlib.pyplot as plt
from utilities_ import run_epoch, run_val, predictmaneuver
from DataLoader import DataLoader
import matplotlib as mpl
from LoadMuldiconProperties import LoadMuldiconProperties

# Directories
#data_directory = '/content/drive/MyDrive/Thesis NLR/data/'
data_directory = 'D:/Thesis/POD_LSTM/data/'
validation_maneuver = 'pitch_a10_A10_f05/'
test_maneuver = 'pitch_a10_A10_f05/'
model_path = 'D:/Thesis/POD_LSTM/Autoencoder/model.pt'
best_model_path = 'D:/Thesis/POD_LSTM/Autoencoder/best_model.pt'

#Options
additional_data = True #Adds sharp pitch up data for training
train_model = True #If false pretrained best model is used for inference

# Define the device to use for training (e.g. GPU, CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the latent dimension for the autoencoder
latent_dim = 20

# Create the models: AE and Hidden MLP
model_AE = ConvolutionalAutoencoder(latent_dim)
model_lat = MLP(5, 512, latent_dim)

# Move the models to the device
model_AE = model_AE.to(device)
model_lat = model_lat.to(device)

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model_AE.parameters(), lr=0.0001, weight_decay=1e-5)
optimizer_lat = optim.Adam(model_lat.parameters(), lr=0.001, weight_decay=1e-5)
batch_size = 32
weight_loss = 0.1

# Import Data and Create Dataset
train_loader, valid_loader, p_mean, p_std, control_mean, control_std = DataLoader(
    data_directory, validation_maneuver, batch_size, additional_data=additional_data)

print('Dataset creation: done \n')

# Training
if train_model:
    state = {}
    state['best_valid_loss'] = float('inf')
    aux_loss = np.empty([6, 1000])  # store loss values
    print('Starting training \n')
    for epoch in range(1000):
        tr_loss, tr_loss_lat, tr_loss_total = run_epoch(
            epoch, model_AE, model_lat, train_loader, criterion, optimizer, optimizer_lat, weight_loss=weight_loss, device=device)
        valid_loss, valid_loss_lat, valid_loss_total = run_val(
            epoch, model_AE, model_lat, valid_loader, criterion, weight_loss=weight_loss, device=device)
        aux_loss[0, epoch] = tr_loss
        aux_loss[1, epoch] = tr_loss_lat
        aux_loss[2, epoch] = tr_loss_total
        aux_loss[3, epoch] = valid_loss
        aux_loss[4, epoch] = valid_loss_lat
        aux_loss[5, epoch] = valid_loss_total

        if valid_loss < state['best_valid_loss']:
            state['best_valid_loss'] = valid_loss_total
            state['epoch'] = epoch
            state['state_dict_AE'] = model_AE.state_dict()
            state['state_dict_lat'] = model_lat.state_dict()
            state['optimizer_AE'] = optimizer.state_dict()
            state['optimizer_lat'] = optimizer_lat.state_dict()

            torch.save(state, model_path)

        print('Train loss AE: {:.6f} \n'.format(tr_loss))
        print('Train loss MLP: {:.6f} \n'.format(tr_loss_lat))
        print('Train loss Tot: {:.6f} \n'.format(tr_loss_total))

        print('Valid loss AE: {:.6f} \n'.format(valid_loss))
        print('Valid loss MLP: {:.6f} \n'.format(valid_loss_lat))
        print('Valid loss Tot: {:.6f} \n'.format(valid_loss_total))

        plt.figure()
        plt.plot(np.arange(epoch), aux_loss[0, :epoch], label='training AE')
        plt.plot(np.arange(epoch), aux_loss[1, :epoch], label='training MLP')
        plt.plot(np.arange(epoch), aux_loss[2, :epoch], label='training total')
        plt.plot(np.arange(epoch), aux_loss[3, :epoch], label='validation AE')
        plt.plot(np.arange(epoch), aux_loss[4, :epoch], label='validation MLP')
        plt.plot(np.arange(epoch),
                 aux_loss[5, :epoch], label='validation total')
        plt.legend()
        plt.yscale('log')
        plt.show()

    print('Finished Training')

# Testing 
pred_test, true_test = predictmaneuver(model_AE, model_lat, best_model_path, data_directory,
                                       test_maneuver, p_mean, p_std, control_mean, control_std, device=device)

#%% Plotting
plt.style.use(['science', 'ieee'])
MULDICON = LoadMuldiconProperties(data_directory)
shape = [113, 65]
k = 100

p_upper_true = true_test[k, int(shape[0] / 2):, :]
p_upper_pred = pred_test[k, int(shape[0] / 2):, :]

# Create figures for pressure and with force coefficient for upper surface
f = plt.figure(figsize=(8, 3))
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))
#ax3 = plt.subplot2grid((1, 3), (0, 2))

vmin = np.min(p_upper_true)
vmax = np.max(p_upper_true)

f.suptitle('Pressure Upper surface,Snapshot N.' + str(k))
im = ax1.contourf(MULDICON.geom.geom_x_upper_norm, MULDICON.geom.geom_y_upper_norm,
                  p_upper_true, 200, vmin=vmin, vmax=vmax, cmap='jet')
ax1.title.set_text('Ground Truth')

im = ax2.contourf(MULDICON.geom.geom_x_upper_norm, MULDICON.geom.geom_y_upper_norm,
                  p_upper_pred, 200, vmin=vmin, vmax=vmax, cmap='jet')
ax2.title.set_text('Prediction')

#im = ax3.contourf(MULDICON.geom.geom_x_upper_norm, MULDICON.geom.geom_y_upper_norm,
                  #np.abs(p_upper_pred-p_upper_true), 200, vmin=0, vmax=1, cmap='jet')
#ax3.title.set_text('Absolute Error')
f.subplots_adjust(right=0.8)
cbar_ax = f.add_axes([0.85, 0.15, 0.02, 0.7])
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = mpl.cm.cool
clb=f.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap='jet'), cax=cbar_ax)
clb.set_label(r'$C_{\bar{p}}$', labelpad=-20, y=1.1, rotation=0)