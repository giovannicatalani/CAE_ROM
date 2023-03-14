# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:41:44 2023

@author: giosp
"""

import unittest
import Model
from DataLoader import DataLoader
import numpy as np

class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
       super(TestModel, self).__init__(*args, **kwargs)    
       train_loader, valid_loader, self.p_mean, self.p_std, self.control_mean, self.control_std = DataLoader('D:/Thesis/POD_LSTM/data/',
                                                                     'pitch_a10_A10_f05/',32)
       for train_images, train_controls in train_loader:
           self.train_image_sample = train_images[0:1]
           self.train_control_sample = train_controls[0:1]
        
    def test_out_dim(self):
        lat_dim = 5
        AE = Model.ConvolutionalAutoencoder(lat_dim)
        y , y_lat = AE(self.train_image_sample.float())
        self.assertEqual(self.train_image_sample.size(dim=2),y.size(dim=2))
        self.assertEqual(self.train_image_sample.size(dim=3),y.size(dim=3))
        self.assertEqual(y_lat.size(dim=1),lat_dim)
    
    def test_inp_load(self):
        self.assertEqual(self.train_image_sample.size(dim=2),np.shape(self.p_mean)[0])
        self.assertEqual(self.train_image_sample.size(dim=3),np.shape(self.p_mean)[1])
        
        
        
if __name__ == '__main__':
    unittest.main()
        
        
        
        