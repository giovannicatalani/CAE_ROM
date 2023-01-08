# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 16:29:22 2022

@author: giosp
"""

import torch.nn as nn

class MLP(nn.Module):
  """
    Initialize the Multilayer Prceptron Module module.
    This model takes as input the control variables and learn the latent
    representation of the CAE
    
    Parameters:
    input size (int): Size of the control variables vector (es. 5)
    hidden size (int): Size of intermediate MLP layers
    output size (int): Size of the latent space (bottleneck) representation.
  """
  def __init__(self, input_size, hidden_size, output_size):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.Tanh()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.relu = nn.Tanh()
    self.fc3 = nn.Linear(hidden_size, output_size)
    

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    return x