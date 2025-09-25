# -*- coding: utf-8 -*-
"""
ANNGLM models 
"""

import os
import re
import logging
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from scipy.optimize import nnls
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
import random 


GRID_DATA = False

class DeltaDNNGLM(nn.Module):
    def __init__(self, input_size, hidden_size, truncation_len=100, num_layers=2, num_weights_mode=0):
        super(DeltaDNNGLM, self).__init__()
        self.hidden_size = hidden_size
        self.truncation_len = truncation_len 
        self.num_weights_mode = num_weights_mode # 0: stim+bias weights, 1: L+R+bias, 2: stim+last+bias 
        if GRID_DATA: 
            self.output_size = 1
        else: 
            if self.num_weights_mode == 0:
                self.output_size = 2
            elif self.num_weights_mode == 1:
                self.output_size = 3 
            elif self.num_weights_mode == 2:
                self.output_size = 3 #4 

        # learning rule coefficients 
        if not GRID_DATA: 
            self.scaling_factor = nn.Parameter(torch.full((self.output_size,), 1e-3)) # to help with initial stability 
        else: 
            self.scaling_factor = nn.Parameter(torch.full((self.output_size,), 1.0))
        
        # Define the DNN structure
        layers = []
        layers.append(nn.Linear(input_size, hidden_size, bias=True))  # First layer from input to hidden
        self.hidden_size = hidden_size
        
        # Add hidden layers 
        for _ in range(num_layers - 1):
            layers.append(nn.ReLU())  # Add activation function between layers
            layers.append(nn.Linear(hidden_size, hidden_size, bias=True))  # Fully connected hidden layers
        
        # Output layer
        layers.append(nn.ReLU())  # Activation before final layer
        layers.append(nn.Linear(hidden_size, self.output_size, bias=True))  # Final layer
        
        # Stack all the layers together
        self.dnn = nn.Sequential(*layers)
    
    def init_hidden(self, x, W0=None):
        batch_size = x.size(0)
        h0 = torch.zeros(batch_size, 1, self.output_size)
        if W0 is not None:
            h0[:,0,:] = W0 
        return h0
    
    def forward(self, x, W, fix_W=False):
        batch_size, seq_len, _ = x.size() 

        # Output list to store the results from each chunk
        outputs = []

        # Process the sequence in chunks of size `truncation_len` to truncate the gradient 
        for t_start in range(0, seq_len, self.truncation_len):
            t_end = min(t_start + self.truncation_len, seq_len) 
            
            out_chunk = []
            for tt in range(t_end-t_start): 
                ii = t_start+tt
                if GRID_DATA:
                    x_tt = x[:,ii,1:] # neglect the 1st input for current stim 
                else:
                    x_tt = torch.cat([W[:,0,:], x[:,ii,1:]], dim=-1) # [1:] to neglect the 1st input for current stim 
                dW = self.scaling_factor * self.dnn(x_tt).unsqueeze(1) # unsqueeze to add the time dimension for formatting purposes 
                if not fix_W: 
                    W = W + dW
                if GRID_DATA: # for the grid_data just output the predicted dW 
                    out_chunk.append(dW)
                else: 
                    out_chunk.append(W)
            out_chunk = torch.cat(out_chunk, dim=1)
            
            # Detach the hidden state after processing each chunk to truncate the gradient
            W = W.detach()
            
            # Collect the chunk outputs (weights and bias)
            outputs.append(out_chunk)

        # Concatenate the outputs from each chunk along the time dimension
        outputs = torch.cat(outputs, dim=1)  

        # # Compute the weighted sum for each time step using the original input
        if self.num_weights_mode == 0:
            weighted_sum = torch.einsum('bti,bti->bt', outputs[:, :, 0:1], x[:,:,0:1]) + outputs[:, :, -1] # "cBoth" plus "bias"
        elif self.num_weights_mode == 1:
            mask_pos = (x[:, :, 0:1] >= 0).float()
            mask_neg = 1.0 - mask_pos  # same as (x < 0).float()
            part_right = torch.einsum('bti,bti->bt', outputs[:, :, 0:1], x[:, :, 0:1] * mask_pos) 
            part_left = torch.einsum('bti,bti->bt', outputs[:, :, 1:2], x[:, :, 0:1] * mask_neg)             
            weighted_sum = part_right + part_left + outputs[:, :, -1] # L+R+bias 
        elif self.num_weights_mode == 2:
            weighted_sum = torch.einsum('bti,bti->bt', outputs[:, :, 0:1], x[:,:,0:1]) + outputs[:, :, -1] + \
                torch.einsum('bti,bti->bt', outputs[:, :, 1:2], x[:,:,1:2])     # stim + bias + past stim 
           
        # Pass the weighted sum through the sigmoid function
        sigmoid_output = torch.sigmoid(weighted_sum).unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)

        return sigmoid_output, dW[:,-1,:], outputs 
    
    
class DeltaRNNGLM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, truncation_len=500, num_weights_mode=0, DNNfirst=True):
        super(DeltaRNNGLM, self).__init__()
        self.truncation_len = truncation_len  
        self.hidden_size = rnn_hidden_size = hidden_size
        self.num_layers = num_layers 
        self.num_rnn_layers = 1  
        self.num_weights_mode = num_weights_mode # 0: stim+bias weights, 1: L+R+bias, 2: stim+last+bias 
        self.DNNfirst = DNNfirst # put DNN before RNN or after 
        if GRID_DATA: 
            self.state_size = 1
        else: 
            if self.num_weights_mode == 0:
                self.state_size = 2
            elif self.num_weights_mode == 1:
                self.state_size = 3 
            elif self.num_weights_mode == 2:
                self.state_size = 3 #4 
        
        # learning rule coefficients, helps with initial stability  
        self.scaling_factor = nn.Parameter(torch.full((self.state_size,), 1e-3))
        
        if num_layers >= 1: 
        
            ## Add DNN structure for feature extraction 
            if self.DNNfirst: 
                layers = []
                layers.append(nn.Linear(input_size, hidden_size, bias=True))  # First layer from input to hidden
                
                # Add hidden layers, try nn.Tanh() or nn.ReLU()
                for _ in range(num_layers - 1):
                    layers.append(nn.ReLU())  # Add activation function between layers
                    layers.append(nn.Linear(hidden_size, hidden_size, bias=True))  # Fully connected hidden layers
                
                # Output layer
                layers.append(nn.ReLU())  # Activation before final layer
                layers.append(nn.Linear(hidden_size, self.state_size, bias=True))  # Final layer
                
                # Stack all the layers together
                self.dnn = nn.Sequential(*layers)
                ## 
        
                # GRU that outputs `input_size + 1` (weights for each input + 1 for the bias term) 
                # RNN takes self.state_size input_size, compressed representation 
                self.gru = nn.GRU(self.state_size, rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True) 
            else: # RNN first 
                self.gru = nn.GRU(input_size, rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
                
                layers = []
                layers.append(nn.Linear(rnn_hidden_size, hidden_size, bias=True))  # First layer from input to hidden
                
                # Add hidden layers, try nn.Tanh() or nn.ReLU()
                for _ in range(num_layers - 1):
                    layers.append(nn.ReLU())  # Add activation function between layers
                    layers.append(nn.Linear(hidden_size, hidden_size, bias=True))  # Fully connected hidden layers
                
                # Output layer
                layers.append(nn.ReLU())  # Activation before final layer
                layers.append(nn.Linear(hidden_size, self.state_size, bias=True))  # Final layer
                
                # Stack all the layers together 
                self.dnn = nn.Sequential(*layers)         
        else:
            self.gru = nn.GRU(input_size, rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True) 
            
        self.fc = nn.Linear(rnn_hidden_size, self.state_size, bias=False) # "cBoth & "bias # input_size + 1)  # Readout size: `input_size + 1`


    def init_hidden(self, x, W0=None):
        batch_size = x.size(0) 
        self.W0 = W0
        return torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size) # need the num_layers dim 
    
    def forward(self, x, h, fix_W=False):
        batch_size, seq_len, _ = x.size()

        # Output list to store the results from each chunk
        outputs = []
        if self.W0 is not None:
            W = self.W0 
        else:             
            pass # W = self.fc(h)[-1] # [-1] to remove the num_layers axis 
        
        # Process the sequence in chunks of size `truncation_len` to truncate the gradient 
        dW = torch.zeros_like(W)
        for t_start in range(0, seq_len, self.truncation_len):
            t_end = min(t_start + self.truncation_len, seq_len) 
            
            out_chunk = []
            for tt in range(t_end-t_start): 
                ii = t_start+tt 
                if GRID_DATA:
                    x_tt = x[:,ii,1:].unsqueeze(1) # [1:] to neglect the 1st input for current stim  
                else:
                    x_tt = torch.cat([W, x[:,ii,1:]], dim=-1).unsqueeze(1) # unsqueeze to add the time dimension for formatting purposes  
                # _, h = self.gru(x_tt, h) 
                if self.DNNfirst:
                    if self.num_layers >= 1: 
                        _, h = self.gru(self.dnn(x_tt[:,0,:]).unsqueeze(1), h)
                    else:
                        _, h = self.gru(x_tt, h)
                    dW = self.fc(h)[-1] * self.scaling_factor # [-1] to remove the num_layers axis 
                else: # RNN first
                    _, h = self.gru(x_tt, h)
                    if self.num_layers >= 1:
                        dW = self.dnn(h[-1]) * self.scaling_factor 
                    else:
                        dW = self.fc(h)[-1] * self.scaling_factor 
                        
                # dW = 0.99*dW + (1-0.99)* self.scaling_factor * self.dnn(x_tt[:,0,:]) # for debugging 
                if not fix_W:
                    W = W + dW 
                if GRID_DATA: # for the grid_data just output the predicted dW 
                    out_chunk.append(dW.unsqueeze(1)) 
                else: 
                    out_chunk.append(W.unsqueeze(1)) # unsqueeze to add the time dimension for formatting purposes 
            out_chunk = torch.cat(out_chunk, dim=1)
            
            # Detach the hidden state after processing each chunk to truncate the gradient
            h = h.detach()
            W = W.detach()
            
            # Collect the chunk outputs (weights and bias)
            outputs.append(out_chunk)

        # Concatenate the outputs from each chunk along the time dimension
        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, seq_len, input_size + 1)

        # # Compute the weighted sum for each time step using the original input
        if self.num_weights_mode == 0:
            weighted_sum = torch.einsum('bti,bti->bt', outputs[:, :, 0:1], x[:,:,0:1]) + outputs[:, :, -1] # "cBoth" plus "bias"
        elif self.num_weights_mode == 1:
            mask_pos = (x[:, :, 0:1] >= 0).float()
            mask_neg = 1.0 - mask_pos  # same as (x < 0).float()
            part_right = torch.einsum('bti,bti->bt', outputs[:, :, 0:1], x[:, :, 0:1] * mask_pos) 
            part_left = torch.einsum('bti,bti->bt', outputs[:, :, 1:2], x[:, :, 0:1] * mask_neg)             
            weighted_sum = part_right + part_left + outputs[:, :, -1] # L+R+bias 
        elif self.num_weights_mode == 2:
            weighted_sum = torch.einsum('bti,bti->bt', outputs[:, :, 0:1], x[:,:,0:1]) + outputs[:, :, -1] + \
                torch.einsum('bti,bti->bt', outputs[:, :, 1:2], x[:,:,1:2])     # stim + bias + past stim  
                
        # Pass the weighted sum through the sigmoid function
        sigmoid_output = torch.sigmoid(weighted_sum).unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)

        return sigmoid_output, dW, outputs 
    