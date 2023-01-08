
# Convolutional AutoEncoder for Unsteady ROM
[![Python](https://img.shields.io/badge/python-3.8-informational)](https://docs.python.org/3/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the implementation of a Deep Learning Framework based on Convolutional Autoencoder (CAE) for the generation of a non-linear and efficient ROM for the prediction of the unsteady aerodynamic simulations. 
The description of the problem and the dataset can be found in the Thesis of G.Catalani **"Machine Learning based local reduced order modeling for the prediction of unsteady aerodynamic loads"** http://resolver.tudelft.nl/uuid:cd5bf762-ab2a-4c9e-8b51-58a173440830 . This model represents an extension of the linear models based on the Proper Orthogonal Decomposition and Neural Networks presented in the thesis.

<img src="https://github.com/giovannicatalani/CAE_ROM/blob/main/Images/readme_cae_rom.png" width="600" />

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Model](#model)
* [Configuration](#configuration)
* [Usage](#usage)
* [Contributing](#contributing)



### Model

During the offline stage (training) of the model, the Autoencoder and the MLP are trained together: the CAE learns the mapping between the Input Images (113,65) to their latent representation and then back to reconstructed images through the decoder, while the MLP learns the mapping from the control inputs (AoA, pitch rate..) to the AE latent vectors.  The cumulative loss function contains two weighed terms: the AE reconstruction error, and the MLP latent vector prediction error.

<img src="https://github.com/giovannicatalani/CAE_ROM/blob/main/Images/Offline_Stage.png" width="1000" />

During the online stage (testing) of the model, the Encoder part of the CAE is discarderd. The control inputs are fed into the MLP, whose output represented the prediction of the latent vector which is then fed into the pretrained decoder. 

<img src="https://github.com/giovannicatalani/CAE_ROM/blob/main/Images/Online_Stage.jpg" width="600" />



### Configuration

The clone the repository:
```shell script
git clone git@github.com:giovannicatalani/CAE_ROM.git
```
The model uses the **PyTorch** library.
To install dependencies with conda:
```shell script
conda env create -f environment.yml
```

### Usage
The model works with input images of size [113,65]. To adapt it to new images it is sufficient to change the convolution parameters of the Autoencoder.
The input variables are the control variables of the wing (Angle of Attack, First Derivative of AoA, Second Derivative AoA, Pitch Rate, First Derivative Pitch Rate)


### Contributing
