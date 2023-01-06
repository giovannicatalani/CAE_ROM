# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:22:19 2023

@author: giosp
"""
import numpy as np
import torch

def DataLoader(data_directory,validation_maneuver,batch_size,additional_data=True):
    
    test_directory = data_directory + 'simulations/' + validation_maneuver
    #Load data (Numpy Arrays)
    #Pressure Fields
    train = np.load(data_directory + 'simulations/training_maneuver/labels_flow.npy')
    valid = np.load(test_directory + '/labels_flow.npy')
    
    #Control Inputs
    train_lat =  np.load(data_directory + 'simulations/training_maneuver/controls_flow.npy')[:,1:]
    valid_lat =  np.load(test_directory + '/controls_flow.npy')[:,1:]
    
    
    if additional_data:
        #Add sharp pitch up with data cleaning
        train_additional = np.load(data_directory + 'simulations/sharp_pitch_up/labels_flow.npy')[337:5:6237,1:]
        train_additional_lat = np.load(data_directory + 'simulations/sharp_pitch_up/controls_flow.npy')[337:5:6237,1:]
        train = np.concatenate((train,train_additional),axis=0)
        train_lat = np.concatenate((train_lat,train_additional_lat),axis=0)
    
        
    #Create Tensor Dataset
    train = torch.from_numpy(train.astype(np.float))
    valid = torch.from_numpy(valid.astype(np.float))
    
    
    train_lat = torch.from_numpy(train_lat.astype(np.float))
    valid_lat = torch.from_numpy(valid_lat.astype(np.float))
   
    
    #Compute Mean and Std Pressure 
    p_mean = train.mean((0))#for each channel
    p_std = train.std((0))  #for each channel
    #Compute Mean and Std Controls
    control_mean = train_lat.mean((0))
    control_std = train_lat.std((0))
    
    #Normalize
    train = (train-p_mean)/p_std #in pressure
    valid = (valid-p_mean)/p_std #in pressure
    
    
    train_lat = (train_lat-control_mean)/control_std #in pressure
    valid_lat = (valid_lat-control_mean)/control_std #in pressure
    
    
    #Add dim to pressure to be processed by AE
    train = train.unsqueeze(1)
    valid = valid.unsqueeze(1)
   
    
    #Define DataLoaders
    train_ds = torch.utils.data.TensorDataset(train,train_lat)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=1, shuffle=True)
    
    valid_ds = torch.utils.data.TensorDataset(valid,valid_lat)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=1, shuffle=True)
    
    return train_loader, valid_loader, p_mean, p_std, control_mean, control_std


def run_epoch(epoch, model,model_lat, train_loader, criterion, optimizer,optimizer_lat, weight_loss=0.5,device='cpu'):
  running_loss_AE = 0 
  running_loss_lat = 0
  running_loss_total = 0
  for i, (inputs, inputs_lat ) in enumerate(train_loader):
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        inputs   = inputs.float()

        inputs_lat = inputs_lat.to(device)
        inputs_lat   = inputs_lat.float()

        # Zero the parameter gradients
        optimizer.zero_grad()
        optimizer_lat.zero_grad()

        # Forward pass
        outputs, outputs_lat = model(inputs)
        outputs_mlp = model_lat(inputs_lat)
        
        loss_AE = criterion(outputs, inputs)
        loss_lat = criterion(outputs_mlp, outputs_lat)

        loss_total = weight_loss * loss_AE + (1-weight_loss) * loss_lat
        loss_total.backward()
        optimizer.step()

        # Backward pass and optimization
        optimizer.step()
        optimizer_lat.step()
   
        # Print statistics
        running_loss_AE += loss_AE.item()
        running_loss_lat += loss_lat.item()
        running_loss_total += loss_total.item()

        
            
  return running_loss_AE/len(train_loader), running_loss_lat/len(train_loader), running_loss_total/len(train_loader)


def run_val(epoch, model,model_lat, val_loader,criterion, weight_loss=0.5, device='cpu'):
  running_loss_AE = 0 
  running_loss_lat = 0
  running_loss_total = 0
  with torch.no_grad():
     for i, (inputs, inputs_lat) in enumerate(val_loader):
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        inputs   = inputs.float()

        inputs_lat = inputs_lat.to(device)
        inputs_lat   = inputs_lat.float()

        # Forward pass
        outputs, outputs_lat = model(inputs)
        outputs_mlp = model_lat(inputs_lat)
        
        loss_AE = criterion(outputs, inputs)
        loss_lat = criterion(outputs_mlp, outputs_lat)
        loss_total = weight_loss * loss_AE + (1-weight_loss) * loss_lat
                

        # Print statistics
        running_loss_AE += loss_AE.item()
        running_loss_lat += loss_lat.item()
        running_loss_total += loss_total.item()
        
            
  return running_loss_AE/len(val_loader), running_loss_lat/len(val_loader), running_loss_total/len(val_loader)

def predictmaneuver(model,model_lat,model_path,data_directory,test_maneuver,p_mean,p_std,control_mean,control_std,device='cpu'):
    # Load the state dict saved with torch.save
    state_dict = torch.load(model_path,map_location=device)
    # Load the state dict into the model
    model.load_state_dict(state_dict['state_dict_AE'])
    model_lat.load_state_dict(state_dict['state_dict_lat'])
    model.eval()
    model_lat.eval()
    
    test_directory = data_directory + 'simulations/' + test_maneuver
    test = np.load(test_directory + '/labels_flow.npy')
    
    test = torch.from_numpy(test.astype(np.float))
    test = test.unsqueeze(1)
    
    #Control Inputs
    test_lat =  np.load(test_directory + '/controls_flow.npy')[:,1:]       
    test_lat = torch.from_numpy(test_lat.astype(np.float))   
    test_lat = (test_lat-control_mean)/control_std #in pressure
   

    pred_test = torch.empty_like(test) 
    with torch.no_grad():
        for k, (test_control) in enumerate(test_lat):

            test_control = test_control.to(device)
            test_control   = test_control.float()
            
            # Feed forward.
            pred_lat = model_lat(test_control)
            x = model.intermediate(pred_lat)
            x = x.reshape(1, 16, 14, 8)
            pred_test[k] = model.decoder(x)
            
            

    pred_test[:,:,:,:] = pred_test[:,:,:,:]*p_std +p_mean  
    pred_test = pred_test[:,0,:,:].cpu().numpy()
    true_test = test[:,0,:,:].cpu().numpy()
    
    return pred_test, true_test