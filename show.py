# at the very least: 
# pass through many prompts in the model? 
# then how to auto-interp/label each neuron? 

# I might want to
''''

    pass dataset through each SAE
    for each neuron, cache the activation levels per image
    query the top-10 activations 
    auto-interp the top-10 activations
        - activation value per img
        - would want to do some sort of filtering
        - do some sort of clustering by activation value, find outliers
        - 
'''
import torch
from sparse_autoencoder.model import Autoencoder, TopK

hidden_dim_size = 512
expansion_f = 64

state_dict = torch.load('/home/minjune/sparse_autoencoder/sae_model_epoch0_step9999.pth')
sae = Autoencoder.from_state_dict(state_dict, strict=False)
