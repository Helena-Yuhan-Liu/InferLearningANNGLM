# -*- coding: utf-8 -*-
"""
Simulated learning data for REINFORCE. 

@author: hyliu
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

import psytrack_learning as psy
from psytrack_learning.simulate_learning import reinforce    
from psytrack_learning.simulate_learning import simulate_learning
from models import DeltaDNNGLM   
from psytrack_learning.plotting_code_tools import plot_grid 


import argparse
parser = argparse.ArgumentParser(description='')
# Setting for T=2000 
parser.add_argument('--save_data', default=True, type=bool, help='save data for plotting?')
parser.add_argument('--num_animals', default=300, type=int, help='number of of simulated animals')
parser.add_argument('--learning_rule_name', default='reinforce', type=str, help='Learning rule simulated')
parser.add_argument('--hidden_size', default=32, type=int, help='Number of hidden units') 
parser.add_argument('--num_layers', default=3, type=int, help='Number of layers in DNNGLM')  
parser.add_argument('--trunc_len', default=500, type=int, help='TBPTT truncation length') 
parser.add_argument('--START', default=0, type=int, help='START trial idx') 
parser.add_argument('--END', default=2001, type=int, help='End trial idx') 
parser.add_argument('--seed', default=99, type=int, help='Random seed')
parser.add_argument('--num_epoch', default=20, type=int, help='number of training fnuepoch')
parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate') 
parser.add_argument('--gamma_decay', default=0.5, type=float, help='learning rate decay rate') 
parser.add_argument('--alpha_pow', default=-6.0, type=float, help='learning rule learning rate power') 
parser.add_argument('--TRAIN_W0', default=True, type=bool, help='Train W0?')
parser.add_argument('--KNOW_W0', default=False, type=bool, help='Know W0?')
# # Setting for Fig. 10 (T=8000) 
# parser.add_argument('--save_data', default=True, type=bool, help='save data for plotting?')
# parser.add_argument('--num_animals', default=1000, type=int, help='number of of simulated animals')
# parser.add_argument('--learning_rule_name', default='reinforce', type=str, help='Learning rule simulated')
# parser.add_argument('--hidden_size', default=64, type=int, help='Number of hidden units') 
# parser.add_argument('--num_layers', default=3, type=int, help='Number of layers in DNNGLM')  
# parser.add_argument('--trunc_len', default=500, type=int, help='TBPTT truncation length') 
# parser.add_argument('--START', default=0, type=int, help='START trial idx') 
# parser.add_argument('--END', default=8001, type=int, help='End trial idx') 
# parser.add_argument('--seed', default=99, type=int, help='Random seed')
# parser.add_argument('--num_epoch', default=40, type=int, help='number of training fnuepoch')
# parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate') 
# parser.add_argument('--gamma_decay', default=0.2, type=float, help='learning rate decay rate') 
# parser.add_argument('--alpha_pow', default=-8.0, type=float, help='learning rule learning rate power') 
# parser.add_argument('--TRAIN_W0', default=True, type=bool, help='Train W0?')
# parser.add_argument('--KNOW_W0', default=False, type=bool, help='Know W0?')
# # Setting for Fig. 2: 
# parser.add_argument('--save_data', default=True, type=bool, help='save data for plotting?')
# parser.add_argument('--num_animals', default=1000, type=int, help='number of of simulated animals')
# parser.add_argument('--learning_rule_name', default='reinforce', type=str, help='Learning rule simulated')
# parser.add_argument('--hidden_size', default=64, type=int, help='Number of hidden units') 
# parser.add_argument('--num_layers', default=3, type=int, help='Number of layers in DNNGLM')  
# parser.add_argument('--trunc_len', default=100, type=int, help='TBPTT truncation length') 
# parser.add_argument('--START', default=0, type=int, help='START trial idx') 
# parser.add_argument('--END', default=501, type=int, help='End trial idx') 
# parser.add_argument('--seed', default=99, type=int, help='Random seed')
# parser.add_argument('--num_epoch', default=8, type=int, help='number of training fnuepoch')
# parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate') 
# parser.add_argument('--gamma_decay', default=0.2, type=float, help='learning rate decay rate') 
# parser.add_argument('--alpha_pow', default=-4.0, type=float, help='learning rule learning rate power') 
# parser.add_argument('--TRAIN_W0', default=True, type=bool, help='Train W0?')
# parser.add_argument('--KNOW_W0', default=True, type=bool, help='Know W0?')
# #   As described in the paper, we used shorter trial sequences for the main-text simulations 
# #   to enable faster runs and allow multiple repetitions for robustness checks. Because these shorter 
# #   sequences contain fewer early trials, estimating the initial weight directly from data becomes unreliable; 
# #   therefore, we initialized the model with the known initial weight in this setting. 
# #   Although the initial policy weights are typically unknown in practice, we also estimated them from the 
# #   psychometric curve and demonstrated successful recovery in Fig. 10 (setting above), as well as in 
# #   the T=2000 simulation setting above.

args = parser.parse_args()
truncation_len = args.trunc_len 
hidden_size = args.hidden_size 

RUN_TEST = True 
KNOW_W0 = args.KNOW_W0    
TRAIN_W0 = (args.TRAIN_W0) and (not KNOW_W0)
print_every = 10  # print loss every 10 iterations

# Set matplotlib defaults from making files editable and consistent in Illustrator 
colors = psy.COLORS
zorder = psy.ZORDER
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = (1,1,1,0)
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'cmu serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 12

# Set the logging level to suppress warnings
logging.getLogger('matplotlib').setLevel(logging.ERROR)
# Alternatively, you can disable font logging specifically
matplotlib.font_manager._log.setLevel(logging.ERROR)

# Set save path for all figures
spath = "./Figures/"  # UPDATE
sim_colors = ["#D81159", "#4357AD", "#EE8434", "#CC3399", "#409091"]

#%%############################################################################
#                       Helper functions 
############################################################################### 

def estimate_W0(stimulus_values, yy):
    ## Estimate W 
    # Estimate p(y=1 | x) for each unique stimulus value
    unique_stimuli = torch.unique(stimulus_values)  # Get unique stimulus values
    p_y_given_x = []

    for x_val in unique_stimuli:
        indices = (stimulus_values == x_val)  # Find samples with this stimulus value
        if indices.sum() > 0:  # Check if there are any samples
            p_y = yy[indices].float().mean()  # Average of y gives p(y=1 | x)
            p_y_given_x.append((x_val.item(), p_y.item()))

    # Convert results to tensors for plotting
    stimuli, probabilities = zip(*p_y_given_x)
    stimuli = torch.tensor(stimuli)
    probabilities = torch.tensor(probabilities)

    # Recover W using logit (sigmoid^{-1}(p))
    epsilon = 1e-2
    probabilities = probabilities.clamp(epsilon, 1 - epsilon)
    logit_probs = torch.log(probabilities / (1 - probabilities))  # Compute logit(p)

    # Construct X matrix for the unique stimuli
    X_unique = torch.stack((stimuli, torch.ones_like(stimuli)), dim=0)  # 2 x num_unique_stimuli

    # Solve for W using torch.linalg.lstsq
    solution = torch.linalg.lstsq(X_unique.T, logit_probs)
    return solution.solution.squeeze() 

criterion = nn.BCELoss() 

#%%############################################################################
#                       Setup simulated data  
###############################################################################

# Parameters
num_epochs = args.num_epoch 
best_lr = args.learning_rate

# Tiny GRU training code for behavior prediction with binary classification
START = args.START
END = args.END

# # Set the seed for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

true_alpha  = np.array([0.0, 2**(args.alpha_pow)]) # learning rate for the learning rule 
true_sigma  = np.array([0.0, 2**(-6.0)]) # not really used, b/c noise will be set to zero in simulate_learning()

# You can change this! 
sim_learning_rule = globals()[args.learning_rule_name] 

# Multiple simulated "animals", each for correspond to a batch 
if RUN_TEST: 
    num_animals = args.num_animals * 2 # if RUN_TEST, generate another set of num_animals animals for testing 
else: 
    num_animals = args.num_animals  
animal_batch_size = 1
plot_batch = args.num_animals - 1 # which animal for plotting 
X_stack = np.ones((num_animals, args.END, 2))
X_stack[:,:,1] = np.random.choice([-2, 2, -1, 1, -1.5, 1.5, -0.5, 0.5, -1.75, 1.75, -1.25, 1.25, \
                    -0.75, 0.75, -0.25, 0.25, 0.], size=(num_animals,args.END)) 
side_stack = (X_stack[:,:,1]>0).astype(int) 
W0_cboth = np.random.choice([-2.], size=(num_animals))
b0 = np.random.choice([0., -1., 1.], size=(num_animals)) 
W0_est_stack = []
for ii in range(num_animals):     
    X = X_stack[ii]
    side = side_stack[ii]
    if ii >= args.num_animals: # test the 2nd half of animals, assume train and test have the same W0 
        W0 = np.array([b0[ii-args.num_animals], W0_cboth[ii-args.num_animals]]) 
    else:            
        W0 = np.array([b0[ii], W0_cboth[ii]]) 
        
    Wsim, y, y_, r, sim_noise = simulate_learning(X=X, side=side, sigma=true_sigma, sigma0=0,
                                           alpha=true_alpha, W0=W0, learning_rule=sim_learning_rule, seed=args.seed) 
    
    # Get W0_est 
    Wsim_swapped = Wsim.T.copy()
    Wsim_swapped[:, [0, 1]] = Wsim.T[:, [1, 0]] # simulate_learning() and DNNGLM() have different ordering of weights 
    if ii < args.num_animals: # for training set, get W0 (as mentioned above, assume train and test have the same W0)
        if KNOW_W0:
            W0_est_stack.append(torch.tensor(Wsim_swapped[0], dtype=torch.float32).unsqueeze(0)) 
        else: 
            num_samples = 100 # number of samples at the beginning used to construct psychometric curve for estimating W0 
            xx = torch.tensor(X[:num_samples,1], dtype=torch.float64) 
            yy = torch.tensor(y[:num_samples], dtype=torch.float64)
            W0_est_ii = torch.tensor(estimate_W0(xx, yy), dtype=torch.float32)
            W0_est_stack.append(W0_est_ii.unsqueeze(0)) 
    
    if ii==plot_batch: 
        weights = {"Wstim": 1, "Wbias": 1}
        fig_Wsim = psy.plot_weights(Wsim[:2][[1, 0],:], weights) 
        ax = fig_Wsim.gca()
        lines = [line for line in ax.get_lines()] 
        ax.legend(lines[:len(weights)], weights.keys()) 

    # stack data across simulated animals 
    if ii==0: 
        choices = torch.tensor(y, dtype=torch.float32).unsqueeze(0) # unsqueeze to create batch dimension             
        y_stack = torch.tensor(y_, dtype=torch.float32).unsqueeze(0) 
        reward_data = torch.tensor(r, dtype=torch.float32).unsqueeze(0)
        stimulus = torch.tensor(X[:,1], dtype=torch.float32).unsqueeze(0)
        side_data = torch.tensor(side, dtype=torch.float32).unsqueeze(0) 
        Wsim_stack = torch.tensor(Wsim_swapped, dtype=torch.float32).unsqueeze(0)
    else: 
        choices = torch.cat([choices, torch.tensor(y, dtype=torch.float32).unsqueeze(0)], dim=0) 
        y_stack = torch.cat([y_stack, torch.tensor(y_, dtype=torch.float32).unsqueeze(0)], dim=0) 
        reward_data = torch.cat([reward_data, torch.tensor(r, dtype=torch.float32).unsqueeze(0)], dim=0) 
        stimulus = torch.cat([stimulus, torch.tensor(X[:,1], dtype=torch.float32).unsqueeze(0)], dim=0) 
        side_data = torch.cat([side_data, torch.tensor(side, dtype=torch.float32).unsqueeze(0)], dim=0) 
        Wsim_stack = torch.cat([Wsim_stack, torch.tensor(Wsim_swapped, dtype=torch.float32).unsqueeze(0)], dim=0)  
    
W0_est_stack = torch.cat(W0_est_stack, dim=0)
mean_W0_b4 = torch.mean(W0_est_stack[:,0]).detach().numpy()
std_W0_b4 = torch.std(W0_est_stack[:,0]).detach().numpy()
mean_b0_b4 = torch.mean(W0_est_stack[:,1]).detach().numpy()
std_b0_b4 = torch.std(W0_est_stack[:,1]).detach().numpy()    
if TRAIN_W0: 
    print(f'W0_stim est: {torch.mean(W0_est_stack[:,0]):.4f} +/- {torch.std(W0_est_stack[:,0]):.4f}')
    print(f'W0_bias est: {torch.mean(W0_est_stack[:,1]):.4f} +/- {torch.std(W0_est_stack[:,1]):.4f}') 
    W0_est_stack = nn.Parameter(W0_est_stack) # Wrap it as a trainable parameter 
 
stimulus = stimulus.unsqueeze(-1) # unsqueeze to create the neuron dimension 
choices = choices.unsqueeze(-1)  
y_stack = y_stack.unsqueeze(-1) 
reward_data = reward_data.unsqueeze(-1)
side_data = side_data.unsqueeze(-1) 

# Stack input data
target = choices[:,1:]    
# same input to learning_rule used as in Zoe's simulate_learning.py 
inputs = torch.cat([stimulus[:,1:], stimulus[:,:-1], torch.ones_like(stimulus[:,:-1]), choices[:,:-1], reward_data[:,:-1], side_data[:,:-1]], dim=-1)
if RUN_TEST:
    for quantity in ['target', 'inputs', 'y_stack', 'choices', 'Wsim_stack', 'stimulus', 'reward_data', 'side_data']:
        locals()[f"test_{quantity}"] = locals()[quantity][int(num_animals/2):,:,:].clone()
        locals()[quantity] = locals()[quantity][:int(num_animals/2),:,:].clone() 
        
    
#%%############################################################################
#                       Training    
###############################################################################

# # Split data for cross-validation, split by time indices 
_, time_steps, input_size = inputs.shape 
assert time_steps % truncation_len == 0, 'truncation length must be divisible by sequence length'

output_size = 2    
input_size = input_size - 1 + output_size # - current stim + num_weights 
seq_len = time_steps
train_inputs = inputs.clone() 

final_model = DeltaDNNGLM(input_size, hidden_size, truncation_len=args.trunc_len, num_layers=args.num_layers)

optimizer = optim.Adam(list(final_model.parameters())+[W0_est_stack], lr=best_lr) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.gamma_decay)
    
for epoch in range(num_epochs): 
    selected_indices = torch.arange(args.num_animals)
    dataset = TensorDataset(train_inputs, target, selected_indices) 
    dataloader = DataLoader(dataset, batch_size=animal_batch_size, shuffle=True)
    
    # Training loop
    for batch_inputs, batch_targets, batch_indices in dataloader: 
        W0_est = W0_est_stack[batch_indices] 
        h = final_model.init_hidden(batch_inputs, W0_est).to(batch_inputs.device)
        batch_outputs, _, glm_weights_batch = final_model(batch_inputs, h)
        
        # Compute loss
        batch_loss = criterion(batch_outputs.reshape(-1), batch_targets.reshape(-1)) 
        
        # Backward pass and optimization step (assuming optimizer is defined)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()   
    
    scheduler.step()

    # Print loss and accuracy for training
    if ((epoch + 1) % print_every == 0) or (epoch==0):
                
        with torch.no_grad(): 
            final_model.eval() # prediction mode to reduce memory demand                 
                
            # # get training stats across all samples 
            W0_est = W0_est_stack                  
            h = final_model.init_hidden(train_inputs, W0_est.detach()).to(inputs.device)
            final_outputs, _, glm_weights_train = final_model(train_inputs, h) 
            final_loss = criterion(final_outputs.reshape(-1), target.reshape(-1) ) 
            
            final_model.train()                    
        
        # Apply sigmoid to get probabilities
        sigmoid_outputs = final_outputs 

        # Convert probabilities to binary predictions (0 or 1)
        predicted = (sigmoid_outputs > 0.5).float()

        # Calculate accuracy on training data 
        correct = (predicted == target).sum().item()
        total = len(target.reshape(-1))
        log_likelihood = -nn.BCELoss()(final_outputs.reshape(-1), target.reshape(-1) ).float().item()
        final_accuracy = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {final_loss.item():.4e}, Per-trial Log-Likelihood: {log_likelihood:.4f}, Accuracy: {final_accuracy * 100:.2f}%') 

# Per-trial LL on simulated weights for sanity check 
log_likelihood_sim = -nn.BCELoss()(y_stack[:,1:,:].reshape(-1), choices[:,1:,:].reshape(-1)).float().item() 
print(f'Per-trial Log-Likelihood of Wsim: {log_likelihood_sim:.4f}')

if TRAIN_W0: 
    mean_W0_after = torch.mean(W0_est_stack[:,0]).detach().numpy()
    std_W0_after = torch.std(W0_est_stack[:,0]).detach().numpy()
    mean_b0_after = torch.mean(W0_est_stack[:,1]).detach().numpy()
    std_b0_after = torch.std(W0_est_stack[:,1]).detach().numpy()
    print(f'W0_stim est: {torch.mean(W0_est_stack[:,0]):.4f} +/- {torch.std(W0_est_stack[:,0]):.4f}')
    print(f'W0_bias est: {torch.mean(W0_est_stack[:,1]):.4f} +/- {torch.std(W0_est_stack[:,1]):.4f}')  
     

#%%############################################################################
#                       Visualize + save data   
###############################################################################   

if RUN_TEST: 
    with torch.no_grad():
        final_model.eval() 
        W0_est = W0_est_stack 
        h = final_model.init_hidden(test_inputs, W0_est.detach()).to(inputs.device)
        test_outputs, _, glm_weights_test = final_model(test_inputs, h)
        sigmoid_test_outputs = test_outputs 
        test_loss = criterion(test_outputs.reshape(-1), test_target.reshape(-1))
        
        predicted = (sigmoid_test_outputs > 0.5).float()
        correct = (predicted == test_choices[:,1:,:]).sum().item()
        total = len(test_choices[:,1:,:].reshape(-1))
        test_accuracy = correct / total
        log_likelihood_test = -nn.BCELoss()(test_outputs, test_choices[:,1:,:].float()).item() # -bce_loss 
        log_likelihood_sim_test = -nn.BCELoss()(test_y_stack[:,1:,:].reshape(-1), test_choices[:,1:,:].reshape(-1)).float().item() 
        print(f'Test Loss: {test_loss.item():.4e}, Log-Likelihood: {log_likelihood_test:.4f}, LL Wsim: {log_likelihood_sim_test:.4f}')

## Get W and dW 
if RUN_TEST: 
    glm_weights = glm_weights_test[plot_batch,:,:] 
    glm_weights_stack = glm_weights_test.detach().numpy() 
    Wsim_stack_ = test_Wsim_stack.detach().numpy()
else: 
    glm_weights = glm_weights_train[plot_batch,:,:] 
    glm_weights_stack = glm_weights_train.detach().numpy() 
    Wsim_stack_ = Wsim_stack.detach().numpy()
Wsim = Wsim_stack_[plot_batch,:,:].T 

delta_Wsim = Wsim[:,1:] - Wsim[:,:-1] 
# delta_Wpsy = Wpsytrack[:,1:] - Wpsytrack[:,:-1] 
glm_Wrec = glm_weights.detach().numpy().T
delta_Wrec = glm_Wrec[:,1:] - glm_Wrec[:,:-1] 

delta_Wsim_stack = Wsim_stack_[:,1:,0] - Wsim_stack_[:,:-1,0]
delta_Wrec_stack = glm_weights_stack[:,1:,0] - glm_weights_stack[:,:-1,0]
    
readin_weights = {"Wstim": 1, "Wbias": 1}
fig_Wrec = psy.plot_weights(glm_weights.detach().numpy().T[:2], readin_weights) 
ax = fig_Wrec.gca()
lines = [line for line in ax.get_lines()] 
ax.legend(lines[:len(readin_weights)], readin_weights.keys())
            

if args.save_data:    
    fsave_prefix = './saved_npy/psytrack_rule_' + args.learning_rule_name + '_alpha_' + str(args.alpha_pow) + '_nr_' + str(hidden_size) + \
        '_lr_' + str(args.learning_rate) + '_ne_' + str(args.num_epoch) + '_gam_' + str(args.gamma_decay) + '_na_' + str(args.num_animals) + '/'  
    os.makedirs(fsave_prefix, exist_ok=True) # Ensure the directory exists 
    
    fig_Wsim.savefig(fsave_prefix + 'fig_Wsim.pdf')
    fig_Wrec.savefig(fsave_prefix + 'fig_Wrec.pdf') 
    
    # Create a dictionary to store tensors
    test_data = {
        "W0_est_stack": W0_est_stack,
        "test_inputs": test_inputs,
        "test_choices": test_choices,
        "test_stimulus": test_stimulus,
        "test_y_stack": test_y_stack,
        "test_Wsim_stack": test_Wsim_stack
    }
    torch.save(test_data, fsave_prefix + "test_data.pth") # Save the dictionary
     
    # Save the trained model 
    torch.save(final_model.state_dict(), fsave_prefix + "final_model.pth")           
    
    # Save hyperparameters 
    args_dict = vars(args)  # Convert Namespace to dictionary
    torch.save(args_dict, fsave_prefix + "args.pth")

    # Save the accuracies 
    fsave = fsave_prefix + 'scores.npz'
    np.savez(fsave, log_likelihood=log_likelihood, log_likelihood_sim=log_likelihood_sim,\
              log_likelihood_test=log_likelihood_test,log_likelihood_sim_test=log_likelihood_sim_test,\
              mean_W0_b4 = mean_W0_b4, std_W0_b4=std_W0_b4, mean_b0_b4 = mean_b0_b4, std_b0_b4=std_b0_b4,\
              mean_W0_after = mean_W0_after, std_W0_after=std_W0_after, mean_b0_after = mean_b0_after, std_b0_after=std_b0_after) 

    # uncomment to plot 
    plot_grid(fsave_prefix, set_ylim=[-0.005, 0.05], set_yticks=[0.,0.025,0.05]) 
    # # plot_grid(fsave_prefix, set_ylim=[-0.005, 0.05], set_yticks=[0.,0.025,0.05], all_on_one=False, small_plot=True) 