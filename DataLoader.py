# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:14:36 2023

@author: giosp
"""

import numpy as np
import torch

# Constants
TRAINING_MANEUVER_FOLDER = 'simulations/training_maneuver/'
SHARP_PITCH_UP_FOLDER = 'simulations/sharp_pitch_up/'
PRESSURE_FIELDS_FILE = 'labels_flow.npy'
CONTROLS_FILE = 'controls_flow.npy'

# Enumerations
COL_PRESSURE = 0
COL_CONTROL_1 = 1

def load_data(data_directory, folder, file):
    return np.load(data_directory + folder + file)

def load_pressure_fields(data_directory, folder):
    return load_data(data_directory, folder, PRESSURE_FIELDS_FILE)

def load_controls(data_directory, folder):
    return load_data(data_directory, folder, CONTROLS_FILE)[:, COL_CONTROL_1:]

def compute_mean_std(data):
    return data.mean((0)), data.std((0))

def normalize(data, mean, std):
    return (data - mean) / std

def DataLoader(data_directory, validation_maneuver, batch_size, additional_data=True):
    VALIDATION_MANEUVER_FOLDER = 'simulations/' + validation_maneuver

    # Load data (Numpy Arrays)
    train_pressure = load_pressure_fields(data_directory, TRAINING_MANEUVER_FOLDER)
    valid_pressure = load_pressure_fields(data_directory, VALIDATION_MANEUVER_FOLDER)

    train_controls = load_controls(data_directory, TRAINING_MANEUVER_FOLDER)
    valid_controls = load_controls(data_directory, VALIDATION_MANEUVER_FOLDER)

    if additional_data:
        # Add sharp pitch up with data cleaning
        train_additional = load_pressure_fields(data_directory, SHARP_PITCH_UP_FOLDER)[337:5:6237, 1:]
        train_additional_controls = load_controls(data_directory, SHARP_PITCH_UP_FOLDER)[337:5:6237]
        train_pressure = np.concatenate((train_pressure, train_additional), axis=0)
        train_controls = np.concatenate((train_controls, train_additional_controls), axis=0)

    # Compute Mean and Std Pressure
    pressure_mean, pressure_std = compute_mean_std(train_pressure)

    # Compute Mean and Std Controls
    control_mean, control_std = compute_mean_std(train_controls)

    # Normalize
    train_pressure = normalize(train_pressure, pressure_mean, pressure_std)
    valid_pressure = normalize(valid_pressure, pressure_mean, pressure_std)
    train_controls = normalize(train_controls, control_mean, control_std)
    valid_controls = normalize(valid_controls, control_mean, control_std)

    # Convert to Tensor
    train_pressure = torch.from_numpy(train_pressure.astype(np.float))
    valid_pressure = torch.from_numpy(valid_pressure.astype(np.float))

    train_controls = torch.from_numpy(train_controls.astype(np.float))
    valid_controls = torch.from_numpy(valid_controls.astype(np.float))

    # Add dim to pressure to be processed by AE
    train_pressure = train_pressure.unsqueeze(1)
    valid_pressure = valid_pressure.unsqueeze(1)

    # Define DataLoaders
    train_ds = torch.utils.data.TensorDataset(train_pressure, train_controls)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=1, shuffle=True)

    valid_ds = torch.utils.data.TensorDataset(valid_pressure, valid_controls)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=1, shuffle=True)

    return train_loader, valid_loader, pressure_mean, pressure_std, control_mean, control_std
