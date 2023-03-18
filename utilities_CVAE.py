# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:48:05 2023

@author: giosp
"""

import numpy as np
import torch
from Model_CVAE import vae_loss
from DataLoader import DataLoader

def run_epoch(epoch, model, train_loader, optimizer, device='cpu'):
  running_loss_VAE = 0 
  running_recon_loss_VAE = 0
  running_kl_loss_VAE = 0
  
  for i, (inputs, inputs_lat ) in enumerate(train_loader):
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        inputs   = inputs.float()
        
        inputs_lat = inputs_lat.to(device)
        inputs_lat   = inputs_lat.float()

        # Zero the parameter gradients
        optimizer.zero_grad()
       
        # Forward pass
        outputs, mu, log_var = model(inputs, inputs_lat)
         # Backward pass and optimization            
        loss_VAE, recon_loss_VAE, kl_loss_VAE = vae_loss(outputs,inputs,mu,log_var)
        loss_VAE.backward()
        optimizer.step()
           
        # Print statistics
        running_loss_VAE += loss_VAE.item()
        running_recon_loss_VAE += recon_loss_VAE.item()
        running_kl_loss_VAE += kl_loss_VAE.item()
            
  return running_loss_VAE/len(train_loader), running_recon_loss_VAE/len(train_loader), running_kl_loss_VAE/len(train_loader)


def run_val(epoch, model, val_loader, device='cpu'):
  running_loss_VAE = 0
  running_recon_loss_VAE = 0
  running_kl_loss_VAE = 0

  with torch.no_grad():
     for i, (inputs, inputs_lat) in enumerate(val_loader):
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        inputs   = inputs.float()
        
        inputs_lat = inputs_lat.to(device)
        inputs_lat   = inputs_lat.float()

        # Forward pass
        outputs, mu, log_var = model(inputs, inputs_lat)
         # Backward pass and optimization            
        loss_VAE, recon_loss_VAE, kl_loss_VAE = vae_loss(outputs,inputs,mu,log_var)
                 
        # Print statistics
        running_loss_VAE += loss_VAE.item()
        running_recon_loss_VAE += recon_loss_VAE.item()
        running_kl_loss_VAE += kl_loss_VAE.item()
      
  return running_loss_VAE/len(val_loader), running_recon_loss_VAE/len(val_loader), running_kl_loss_VAE/len(val_loader)

def predictmaneuver(model,model_path,data_directory,test_maneuver,p_mean,p_std,control_mean,control_std,device='cpu'):
    # Load the state dict saved with torch.save
    state_dict = torch.load(model_path,map_location=device)
    # Load the state dict into the model
    model.load_state_dict(state_dict['state_dict_CVAE'])
    model.eval()
   
    
    test_directory = data_directory + 'simulations/' + test_maneuver
    test = np.load(test_directory + '/labels_flow.npy')
    
    test = torch.from_numpy(test.astype(np.float))
    test = test.unsqueeze(1)
    
    #Control Inputs
    test_lat =  np.load(test_directory + '/controls_flow.npy')[:,1:]       
    test_lat = torch.from_numpy(test_lat.astype(np.float))   
    test_lat = (test_lat-control_mean)/control_std #in pressure
    
    z = torch.randn(len(test_lat), model.lat_dim, device=device)
   
    pred_test = torch.empty_like(test) 
    with torch.no_grad():
        for k, (test_control) in enumerate(test_lat):

            test_control = test_control.to(device)
            test_control   = test_control.float()
            
            print(test_control.size())
            print(z[k].size())
            pred_test[k] = model.decode(z[k].unsqueeze(0), test_control.unsqueeze(0))
            
            

    pred_test[:,:,:,:] = pred_test[:,:,:,:]*p_std +p_mean  
    pred_test = pred_test[:,0,:,:].cpu().numpy()
    true_test = test[:,0,:,:].cpu().numpy()
    
    return pred_test, true_test

def generatesample(model, model_path, p_mean, p_std, num_samples, device='cpu'):
    
    # Load the state dict saved with torch.save
    state_dict = torch.load(model_path, map_location=device)
    # Load the state dict into the model
    model.load_state_dict(state_dict['state_dict_VAE'])
    model.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, model.lat_dim).to(device) # Sample from standard normal distribution
        samples = model.decode(z) # Decode the samples
    
    samples = samples * p_std + p_mean # Rescale the samples to original range
    samples = samples[:, 0, :, :].cpu().numpy() # Convert to numpy array and extract first channel
    
    return samples

def reconstruct(model,model_path,data_directory,test_maneuver,p_mean,p_std,device='cpu'):
    # Load the state dict saved with torch.save
    state_dict = torch.load(model_path,map_location=device)
    # Load the state dict into the model
    model.load_state_dict(state_dict['state_dict_VAE'])
    model.eval()
    
    test_directory = data_directory + 'simulations/' + test_maneuver
    true_test = np.load(test_directory + '/labels_flow.npy')
    test = (true_test-p_mean)/p_std
    
    test = torch.from_numpy(test.astype(np.float))
    test = test.unsqueeze(1)
    
    pred_test = []
    mu_test = []
    log_var_test = []
    
    batch_size = 10
    with torch.no_grad():
        for i in range(len(test) // batch_size):
            test_batch = test[batch_size*i:batch_size*(i+1)].to(device).float()
            
            # Feed forward.
            pred, mu, log_var = model(test_batch)
            pred = pred[:,0,:,:]*p_std +p_mean  
            
            pred_test.append(pred.cpu().numpy())
            mu_test.append(mu.cpu().numpy())
            log_var_test.append(log_var.cpu().numpy())
            
    pred_test = np.concatenate(pred_test, axis=0)
    mu_test = np.concatenate(mu_test, axis=0)
    log_var_test = np.concatenate(log_var_test, axis=0)
    
        
    return pred_test, true_test, mu_test, log_var_test


    
