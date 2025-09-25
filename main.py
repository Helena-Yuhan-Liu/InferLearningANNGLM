# -*- coding: utf-8 -*-
"""
Running ANNGLM on IBL data (as used in Ashwood et al., NeurIPS'20)
"""
import os
import re
import logging
import matplotlib
from matplotlib import pyplot as plt 
from scipy.optimize import minimize
from scipy.special import expit
from scipy.optimize import nnls
# from sklearn.linear_model import Lasso
# from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
import random 

import psytrack_learning as psy
from psytrack_learning.getMAP import getMAP
from psytrack_learning.helper.helperFunctions import update_hyper, hyper_to_list
from psytrack_learning.helper.jacHessCheck import compHess, compHess_nolog
from psytrack_learning.helper.invBlkTriDiag import getCredibleInterval
from psytrack_learning.hyperparameter_optimization import evd_lossfun
from psytrack_learning.learning_rules import RewardMax, PredictMax, REINFORCE, REINFORCE_base
from psytrack_learning.simulate_learning import reward_max, predict_max, reinforce, reinforce_base
from psytrack_learning.simulate_learning import simulate_learning
from models import DeltaDNNGLM, DeltaRNNGLM 

import argparse
parser = argparse.ArgumentParser(description='') 
parser.add_argument('--save_data', default=True, type=bool, help='save data for plotting?')
parser.add_argument('--animal_name', default='CSHL_003', type=str, help='Name of the mouse to analyze')
parser.add_argument('--glmw_mode', default=1, type=int, help='0: DeltaRNNGLM; 1: DeltaDNNGLM')
parser.add_argument('--num_epoch', default=1000, type=int, help='number of training fnuepoch')
parser.add_argument('--hidden_size', default=32, type=int, help='Number of hidden units') 
parser.add_argument('--num_layers', default=2, type=int, help='Number of layers') 
parser.add_argument('--num_dnnrnn_layers', default=2, type=int, help='Number of DNN layers b4 RNN') 
parser.add_argument('--trunc_len', default=500, type=int, help='TBPTT truncation length') 
parser.add_argument('--END', default=10001, type=int, help='End trial idx') 
parser.add_argument('--folds', default=5, type=int, help='Number of cross validation folds')
parser.add_argument('--seed', default=99, type=int, help='Random seed')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate') 
parser.add_argument('--gamma_decay', default=0.999, type=float, help='learning rate decay rate') 
parser.add_argument('--num_weights_mode', default=0, type=int, help='0: stim+bias weights, 1: L+R+bias, 2: stim+bias+history')

args = parser.parse_args()
animal_name = args.animal_name
truncation_len = args.trunc_len 
hidden_size = args.hidden_size 

TRAIN_ONLY = True # False for cross-validation 
RUN_TEST = False # True for test data at the end 

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
#                       Helper functions + loading data  
############################################################################### 

# Mouse data 
mouse_data_path = spath + "ibl_processed.csv"   # --- UPDATE if necessary ---
MOUSE_DF = pd.read_csv(mouse_data_path)

def getMouse(subject, p=5):
    df = MOUSE_DF[MOUSE_DF['subject']==subject]   # Restrict data to the subject specified
    
    cL = np.tanh(p*df['contrastLeft'])/np.tanh(p)   # tanh transformation of left contrasts
    cR = np.tanh(p*df['contrastRight'])/np.tanh(p)  # tanh transformation of right contrasts
    cBoth = cR - cL
    inputs = dict(cL = np.array(cL)[:, None], cR = np.array(cR)[:, None], cBoth = np.array(cBoth)[:, None])

    dat = dict(
        subject=subject,
        lab=np.unique(df["lab"])[0],
        contrastLeft=np.array(df['contrastLeft']),
        contrastRight=np.array(df['contrastRight']),
        date=np.array(df['date']),
        dayLength=np.array(df.groupby(['date','session']).size()),
        correct=np.array(df['feedbackType']),
        answer=np.array(df['answer']),
        probL=np.array(df['probabilityLeft']),
        inputs = inputs,
        y = np.array(df['choice'])
    )
    
    return dat 

def aggregate_unique_x(x, y):
    unique_x, indices = np.unique(x, return_inverse=True)
    mean_y = np.zeros_like(unique_x, dtype=float)
    std_y = np.zeros_like(unique_x, dtype=float)
    
    for i, ux in enumerate(unique_x):
        mask = (x == ux)
        mean_y[i] = np.mean(y[mask])
        std_y[i] = np.std(y[mask], ddof=1)  # Using ddof=1 for sample std deviation
    
    return unique_x, mean_y, std_y

def compute_reward_condition(rr_stack, S=5, correct_cond=True):
    """
    Compute a boolean array `reward_cond` of the same shape as `rr_stack`, 
    where each entry is True if all previous S timesteps (or as many as available)
    had rr_stack == 1, and False otherwise.
    
    Parameters:
    - rr_stack (ndarray): A numpy array of shape (batch, time) with float entries (0.0 or 1.0)
    - S (int): Number of previous timesteps to consider (default=5)

    Returns:
    - reward_cond (ndarray): A boolean array of shape (batch, time)
    """
    batch_size, time_steps = rr_stack.shape
    reward_cond = np.zeros((batch_size, time_steps), dtype=bool)  # Initialize output array

    for t in range(1, time_steps):
        # Consider previous S timesteps (or as many as available)
        start_idx = max(0, t - S) 
        if correct_cond: 
            reward_cond[:, t] = np.all(rr_stack[:, start_idx:t] > 0.5, axis=1) 
        else:
            reward_cond[:, t] = np.all(rr_stack[:, start_idx:t] < 0.5, axis=1)             

    return reward_cond
    

#%%############################################################################
#                       Setup data  
###############################################################################

# Parameters
learning_rate = args.learning_rate
num_epochs = args.num_epoch 
print_every = 10  # print loss every 30 iterations
folds = args.folds

# Tiny GRU training code for behavior prediction with binary classification
START = 0
END = args.END

# # Set the seed for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Now stack data 
jj=0 
animal_list = ['CSHL_001', 'CSHL_002', 'CSHL_003', 'CSHL_004', 'CSHL_005', 'CSHL_006', 'CSHL_007',
                'CSHL_008', 'CSHL_010', 'CSHL_012', 'CSHL_014', 'CSHL_015']
animal_batch_size = len(animal_list) 
for animal_ii in animal_list: 
    orig_dat = getMouse(animal_ii, 5)
    seg_dat = psy.trim(orig_dat, START=START, END=END)
    
    stimulus_ii = np.squeeze(seg_dat['inputs']['cBoth'])  # shape (5000,)
    day_start_ii = np.zeros_like(stimulus_ii)
    try:
        day_start_ii[np.cumsum(seg_dat['dayLength']).astype(int)] = 1 
    except: 
        day_start_ii[np.cumsum(seg_dat['dayLength'][:-1]).astype(int)] = 1 
    choices_ii = seg_dat['y']  # shape (5000,)
    reward_data_ii = (seg_dat['answer'] == seg_dat['y']).astype(float) # shape (5000,)
    answer_data_ii = seg_dat['answer'] # unsqueeze to create the batch dimension!! 
    
    if jj==0:
        stimulus = stimulus_ii[None,:] 
        day_start = day_start_ii[None,:]
        choices = choices_ii[None,:]
        reward_data = reward_data_ii[None,:]
        answer_data = answer_data_ii[None,:]
    else:
        stimulus = np.concatenate((stimulus, stimulus_ii[None,:]), axis=0)
        day_start = np.concatenate((day_start, day_start_ii[None,:]), axis=0)
        choices = np.concatenate((choices, choices_ii[None,:]), axis=0)
        reward_data = np.concatenate((reward_data, reward_data_ii[None,:]), axis=0)
        answer_data = np.concatenate((answer_data, answer_data_ii[None,:]), axis=0)
    jj+=1 
    
# Convert numpy arrays to PyTorch tensors 
stimulus = torch.tensor(stimulus, dtype=torch.float32).unsqueeze(-1) # unsqueeze to create the neuron dimension 
choices = torch.tensor(choices, dtype=torch.float32).unsqueeze(-1)  
day_start = torch.tensor(day_start, dtype=torch.float32).unsqueeze(-1)
reward_data = torch.tensor(reward_data, dtype=torch.float32).unsqueeze(-1)
answer_data = torch.tensor(answer_data, dtype=torch.float32).unsqueeze(-1) 

# Stack input data
# Input positions correspond to how DNN/RNNGLM is coded, some input will be used for the GLM regression and learning rule function 
inputs = torch.cat([stimulus[:,1:], stimulus[:,:-1], torch.ones_like(stimulus[:,:-1]), day_start[:,:-1], choices[:,:-1], reward_data[:,:-1], answer_data[:,:-1]], dim=-1) 
target = choices[:,1:] # target = choices[1:] # Target is binary, shape (4999, 1) 
    
#%%############################################################################
#                       Training    
###############################################################################

# Loss function definition 
criterion = nn.BCELoss() 

batch_size, time_steps, input_size = inputs.shape 

# Split by batch/animal indicies for cross-validation  
# Shuffle indices randomly
indices = np.random.permutation(batch_size)
test_indices = np.array_split(indices, folds)
test_indices = [list(test_set) for test_set in test_indices] 
test_scores = []
train_scores = [] 

assert time_steps % truncation_len == 0, 'truncation length must be divisible by sequence length'
if args.num_weights_mode == 0:
    output_size = 2
elif args.num_weights_mode == 1:
    output_size = 3 
elif args.num_weights_mode == 2:
    output_size = 3 #4 
input_size = input_size - 1 + output_size
seq_len = time_steps
glm_weights_all = torch.zeros(batch_size,seq_len, output_size) 

for ff in range(folds):
    
    if TRAIN_ONLY: 
        train_inputs = inputs.clone() 
        train_target_mask = torch.ones(batch_size)
        
    else:    
        print(f'#### fold={ff} ####')
        # Create train and test masks 
        test_idx = test_indices[ff]
        test_idx.sort() 
        train_target_mask = torch.ones(batch_size)
        test_target_mask = torch.zeros(batch_size) 
        train_target_mask[test_idx] = 0 
        test_target_mask[test_idx] = 1
    
        val_inputs = inputs.clone() # copy all inputs for now, to be masked later 
        train_inputs = inputs.clone() # copy all inputs for now, to be masked later 

    # Instantiate the model
    if args.glmw_mode == 1:
        final_model = DeltaDNNGLM(input_size, hidden_size, truncation_len=args.trunc_len, num_layers=args.num_layers, num_weights_mode=args.num_weights_mode)
    elif args.glmw_mode == 0:
        final_model = DeltaRNNGLM(input_size, hidden_size, num_layers=args.num_dnnrnn_layers, truncation_len=args.trunc_len, num_weights_mode=args.num_weights_mode, DNNfirst=False)
            
    optimizer = optim.Adam(final_model.parameters(), lr=learning_rate) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.gamma_decay)

    for epoch in range(num_epochs):
        # Create a dataset and dataloader for SGD
        selected_indices = torch.nonzero(train_target_mask.flatten() == 1, as_tuple=False).squeeze()
        dataset = TensorDataset(train_inputs[train_target_mask == 1,:,:], target[train_target_mask == 1,:,:], selected_indices) 
        dataloader = DataLoader(dataset, batch_size=animal_batch_size, shuffle=True)
        
        # Training loop
        for batch_inputs, batch_targets, batch_indices in dataloader: # for batch_inputs, batch_targets, batch_Wsim, batch_indices in dataloader:
            W0_est = torch.zeros((batch_inputs.size(0), output_size)) # assume both IBL glm weights starts at W0=0, as in Ashwood et al. 
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
                W0_est = torch.zeros((train_inputs[train_target_mask == 1,:,:].shape[0], output_size))      
                h = final_model.init_hidden(train_inputs[train_target_mask == 1,:,:], W0_est.detach()).to(inputs.device)
                final_outputs, _, glm_weights_train = final_model(train_inputs[train_target_mask == 1,:,:], h) 
                final_loss = criterion(final_outputs.reshape(-1), target[train_target_mask == 1,:,:].reshape(-1) ) 
                
                final_model.train()                    
            
            # Apply sigmoid to get probabilities
            sigmoid_outputs = final_outputs 

            # Convert probabilities to binary predictions (0 or 1)
            predicted = (sigmoid_outputs > 0.5).float()

            # Calculate accuracy on training data 
            correct = (predicted == target[train_target_mask == 1,:,:]).sum().item()
            total = len(target[train_target_mask == 1,:,:].reshape(-1))
            log_likelihood = -nn.BCELoss()(final_outputs.reshape(-1), \
                target[train_target_mask == 1,:,:].reshape(-1) ).float().item() 
            final_accuracy = correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {final_loss.item():.4e}, Log-Likelihood: {log_likelihood:.4f}, Accuracy: {final_accuracy * 100:.2f}%') 
        
    # Validation 
    if not TRAIN_ONLY: 
        with torch.no_grad(): 
            final_model.eval()
            W0_est = torch.zeros((inputs[test_target_mask == 1,:,:].shape[0], output_size))    
            
            shuffle_time_indices = torch.randperm(inputs.size(1)) 
            val_inputs = inputs[test_target_mask==1,:,:] #[:, shuffle_time_indices, :]
            val_target = target[test_target_mask == 1,:,:] #[:, shuffle_time_indices, :]
            h = final_model.init_hidden(val_inputs, W0_est.detach()).to(val_inputs.device)
            test_outputs, _, glm_weights_fold = final_model(val_inputs, h)  
            sigmoid_test_outputs = test_outputs 
            
            test_loss = criterion(test_outputs.reshape(-1), val_target.reshape(-1))                 
            predicted = (sigmoid_test_outputs > 0.5).float()
            correct = (predicted == val_target).sum().item()
            total = len(val_target.reshape(-1))
            test_accuracy = correct / total
            log_likelihood_test = -nn.BCELoss()(test_outputs.reshape(-1), \
                                     val_target.reshape(-1).float()).item() # -bce_loss 
            print(f'Val Loss: {test_loss.item():.4e}, Log-Likelihood: {log_likelihood_test:.4f}, Accuracy: {test_accuracy * 100:.2f}%') 
            glm_weights_all[test_target_mask==1,:,:] = glm_weights_fold 
            
            # Save plot 
            val_mask = (test_target_mask == 1).numpy()
            val_animals = [animal for animal, mask in zip(animal_list, val_mask) if mask]
            for plot_batch in range(len(val_animals)): 
                glm_weights = glm_weights_fold[plot_batch,:,:]
                if args.num_weights_mode == 0:
                    readin_weights = {"Wstim": 1, "Wbias": 1}
                elif args.num_weights_mode == 1:
                    readin_weights = {"cR": 1, "cL": 1, "Wbias": 1} 
                elif args.num_weights_mode == 2:
                    readin_weights = {"Wstim": 1, "h": 1, "Wbias": 1}
                
                fsave_prefix = './Figures/saved_npy/psytrack_animal_' + val_animals[plot_batch] + '_glmw_' + str(args.glmw_mode) + '_nl_' + str(args.num_layers) + '_nr_' + str(hidden_size) + '_lr_' + str(args.learning_rate) + '_ne_' + str(args.num_epoch) + '_wm_' + str(args.num_weights_mode)                
                fig_Wrec = psy.plot_weights(glm_weights.detach().numpy().T, readin_weights) 
                ax = fig_Wrec.gca() 
                lines = [line for line in ax.get_lines()] 
                ax.legend(lines[:len(readin_weights)], readin_weights.keys()) 
                fig_Wrec.savefig(fsave_prefix + '_valWrec.pdf') 
                np.save(fsave_prefix + '_valWrec.npy', glm_weights.detach().numpy().T)
                          
            final_model.train()
            
        test_scores.append(log_likelihood_test)
        train_scores.append(log_likelihood)
        
    else: # just need loop through once if only training 
        print("Breaking out of the cross-val loop for training only")
        break 
    

#%%############################################################################
#                       Print results   
###############################################################################

if not TRAIN_ONLY: 
    mean_train_score = np.mean(train_scores)
    std_train_score = np.std(train_scores)
    print(f'Average Train LL: {mean_train_score:.4f}, +/- {std_train_score:.4f}')
    
    mean_test_score = np.mean(test_scores)
    std_test_score = np.std(test_scores)
    print(f'Average Val LL: {mean_test_score:.4f}, +/- {std_test_score:.4f}')
        
if RUN_TEST: # Get test data from future TEST_LEN trials  
    TEST_LEN = 500 
    jj=0 
    for animal_ii in animal_list: 
        orig_dat = getMouse(animal_ii, 5)
        seg_dat = psy.trim(orig_dat, START=START, END=(END+TEST_LEN))
        
        stimulus_ii = np.squeeze(seg_dat['inputs']['cBoth'])  # shape (5000,)
        day_start_ii = np.zeros_like(stimulus_ii)
        try:
            day_start_ii[np.cumsum(seg_dat['dayLength']).astype(int)] = 1 
        except: 
            day_start_ii[np.cumsum(seg_dat['dayLength'][:-1]).astype(int)] = 1 
        choices_ii = seg_dat['y']  # shape (5000,)
        reward_data_ii = (seg_dat['answer'] == seg_dat['y']).astype(float) # shape (5000,)
        answer_data_ii = seg_dat['answer'] # unsqueeze to create the batch dimension!! 
        
        if jj==0:
            test_stimulus = stimulus_ii[None,:] 
            test_day_start = day_start_ii[None,:]
            test_choices = choices_ii[None,:]
            test_reward_data = reward_data_ii[None,:]
            test_answer_data = answer_data_ii[None,:]
        else:
            test_stimulus = np.concatenate((test_stimulus, stimulus_ii[None,:]), axis=0)
            test_day_start = np.concatenate((test_day_start, day_start_ii[None,:]), axis=0)
            test_choices = np.concatenate((test_choices, choices_ii[None,:]), axis=0)
            test_reward_data = np.concatenate((test_reward_data, reward_data_ii[None,:]), axis=0)
            test_answer_data = np.concatenate((test_answer_data, answer_data_ii[None,:]), axis=0)
        jj+=1 

    test_stimulus = torch.tensor(test_stimulus, dtype=torch.float32).unsqueeze(-1) # unsqueeze to create the neuron dimension 
    test_choices = torch.tensor(test_choices, dtype=torch.float32).unsqueeze(-1)  
    test_day_start = torch.tensor(test_day_start, dtype=torch.float32).unsqueeze(-1)
    test_reward_data = torch.tensor(test_reward_data, dtype=torch.float32).unsqueeze(-1)
    test_answer_data = torch.tensor(test_answer_data, dtype=torch.float32).unsqueeze(-1) 
    test_inputs = torch.cat([test_stimulus[:,1:], test_stimulus[:,:-1], torch.ones_like(test_stimulus[:,:-1]), \
                             test_day_start[:,:-1], test_choices[:,:-1], test_reward_data[:,:-1], test_answer_data[:,:-1]], dim=-1) 
    test_target = test_choices[:,1:] 
    
    # Test the model 
    final_model.eval() 
    W0_est = torch.zeros((test_inputs.shape[0], output_size)) 
    h = final_model.init_hidden(test_inputs, W0_est.detach()).to(inputs.device) 
    test_outputs, _, _ = final_model(test_inputs, h)  
    log_likelihood_test = -nn.BCELoss()(test_outputs[:,-TEST_LEN:].reshape(-1), \
                             test_target[:,-TEST_LEN:].reshape(-1).float()).item() 
    print(f'Test Log-Likelihood: {log_likelihood_test:.4f}')   

#%%############################################################################
#                       Plot results   
###############################################################################     

## Plot dW function 
# Setup grid input data 
cond_past = True # condition on past trials? You can change this 
SS=3 # number of past trials to condition on... You can change this 

set_ylim = [-0.05, 0.05] 
x_val = np.array([-1.        , -0.98670389, -0.84836067, -0.55465008, -0.30273722, 0.,
        0.30273722,  0.55465008,  0.84836067,  0.98670389, 1.        ]) # different x's found in IBL data 
y_val = np.array([0., 1.]) # np.array([0., 1.])
w_val = np.array([0., 1., 2.])
b_val = np.array([-1.0, 0., 1.0]) 

y1_val = np.array([0., 1.])
y2_val = np.array([0., 1.])
y3_val = np.array([0., 1.])
y4_val = np.array([0., 1.])
x1_val = np.array([-1., 1.])
x2_val = np.array([-1., 1.])
x3_val = np.array([-1., 1.])
x4_val = np.array([-1., 1.])

# Create the grid and flatten it
X_, Y_, W_, B_, X1_, Y1_, X2_, Y2_, X3_, Y3_, X4_, Y4_ = np.meshgrid(x_val, y_val, w_val, b_val, \
                             x1_val, y1_val, x2_val, y2_val, x3_val, y3_val, x4_val, y4_val, indexing='ij')
stimulus = np.concatenate((X1_.ravel()[:,None], X2_.ravel()[:,None], X3_.ravel()[:,None], X4_.ravel()[:,None], X_.ravel()[:,None]),axis=-1)
choices = np.concatenate((Y1_.ravel()[:,None], Y2_.ravel()[:,None], Y3_.ravel()[:,None], Y4_.ravel()[:,None], Y_.ravel()[:,None]),axis=-1)
w_vals = W_.ravel()
b_vals = B_.ravel() 

# Convert to torch tensors in float type
stimulus = torch.tensor(stimulus, dtype=torch.float32).unsqueeze(-1) # add unit dimension 
choices = torch.tensor(choices, dtype=torch.float32).unsqueeze(-1)
w_vals = torch.tensor(w_vals, dtype=torch.float32).unsqueeze(-1)
b_vals = torch.tensor(b_vals, dtype=torch.float32).unsqueeze(-1)
w_stack = torch.cat([w_vals, b_vals], axis=-1)

# Define answer and reward_data based on conditions
answer_data = (stimulus > 0).float()
reward_data = (choices == answer_data).float()
day_start = torch.zeros_like(choices) 
inputs_stack = torch.cat([stimulus, stimulus, torch.ones_like(stimulus), day_start, choices, reward_data, answer_data], dim=-1)

# Get dW and plot
W0_est = w_stack 
h = final_model.init_hidden(inputs_stack, W0_est).to(inputs_stack.device) # To change? 
_, dW, _ = final_model(inputs_stack, h, fix_W=True) # fix_W=True: use W0 for W_{T-1}, in order to accurately reflect the input weight going into the function 
delta_Wrec_stack = dW[:,0:1].detach().numpy()  
Wrec_stack = w_vals.detach().numpy()
brec_stack = b_vals.detach().numpy()

# Plot + fit 
xx_ = stimulus[:,-1].detach().numpy()
yy_ = choices[:,-1].detach().numpy()
ww_ = w_vals.detach().numpy()
bb_ = b_vals.detach().numpy() 
zz_ = answer_data[:,-1].detach().numpy() 
rr_ = reward_data[:,-1].detach().numpy() 
    
eps=0.01  
bref_list = [0, -1, 1] 
wref_list = [2, 0, 1]
dW_poscond_dict = {'correct':{'2':[],'0':[],'1':[]}, 'incorrect':{'2':[],'0':[],'1':[]}}
dW_negcond_dict = {'correct':{'2':[],'0':[],'1':[]}, 'incorrect':{'2':[],'0':[],'1':[]}}
wcolor_list = ['m', 'g', 'b']
wref_list_plot = wref_list.copy()

for past_correct_cond in [True, False]: 
    if cond_past: 
        rew_cond_mask_ = compute_reward_condition(reward_data[:,:,0].detach().numpy(), S=SS, correct_cond=past_correct_cond) 
        rew_cond_mask_ = rew_cond_mask_[:,-1:] # only comparing at the last time step

    fig_dWstack_heat_wslices = plt.figure(figsize=(12, 4))        
    plot_ii = 0
    for bref in bref_list: #[0]: 
            
        def plot_wslices(wref, deltaW, color, correct=True): 
            if correct:
                mask = (np.abs(ww_ - wref) < eps) & ((yy_ > 0.5) == (xx_ > 0.)) & (np.abs(bb_ - bref) < eps) 
            else: 
                mask = (np.abs(ww_ - wref) < eps) & ((yy_ > 0.5) != (xx_ > 0.)) & (np.abs(bb_ - bref) < eps) 
            mask = (mask & rew_cond_mask_) if cond_past else mask 
            x_smooth, y_smooth, _ = aggregate_unique_x(xx_[mask], deltaW[mask])
            wlabel = f"W={wref}{', correct' if correct else ', incorrect'}" 
            if correct:
                plt.plot(x_smooth, y_smooth, label=wlabel, color=color, marker='.', linestyle='-', linewidth=0.5) 
            else: 
                plt.plot(x_smooth, y_smooth, label=wlabel, color=color, linestyle='--') 
            plt.xlabel(r'$\mathbf{X}$', fontsize=16)
            plt.ylabel(r'$\mathbf{\Delta W}$', fontsize=16)               
            plt.ylim(set_ylim)         
            plt.legend() #(fontsize=LEGEND_FONT) 
            return x_smooth, y_smooth

        plt.subplot(1, len(bref_list), plot_ii+1)
        for wref, wcolor in zip(wref_list, wcolor_list):
            x_smooth, y_smooth = plot_wslices(wref, delta_Wrec_stack, wcolor, correct=True)
            if bref==0: # only save for bias=0 
                if past_correct_cond:
                    dW_poscond_dict['correct'][str(wref)] = y_smooth 
                else:
                    dW_negcond_dict['correct'][str(wref)] = y_smooth 
            x_smooth, y_smooth = plot_wslices(wref, delta_Wrec_stack, wcolor, correct=False)
            if bref==0: # only save for bias=0 
                if past_correct_cond:
                    dW_poscond_dict['incorrect'][str(wref)] = y_smooth 
                else:
                    dW_negcond_dict['incorrect'][str(wref)] = y_smooth 
        if args.glmw_mode==0: 
            plt.title('Wbias = ' + str(bref) + '\nRNNGLM', fontsize=20)
        elif args.glmw_mode==1:
            plt.title('Wbias = ' + str(bref) + '\nDNNGLM', fontsize=20)
        
        plot_ii += 1              
    plt.tick_params(axis='both', which='major', labelsize=12)  # Set font size for axis ticks
    plt.tight_layout(pad=2.0)
    plt.show() 

if cond_past: 
    fig_Wrec = plt.figure(figsize=(4,3)) #plt.figure(figsize=(12,4))
    wref_list = [0] # wref_list.sort() 
    for ww in range(len(wref_list)):
        wref = wref_list[ww]
        plt.subplot(1, len(wref_list), ww+1)
        plt.plot(x_smooth, dW_poscond_dict['correct'][str(wref)], color='k', marker='.', linestyle='-', linewidth=0.5, label='Past S=3 trials rewarded')
        plt.plot(x_smooth, dW_negcond_dict['correct'][str(wref)], color='r', marker='.', linestyle='-', linewidth=0.5, label='Past S=3 trials unrewarded')
        plt.xlabel(r'$\mathbf{X}$', fontsize=16)
        plt.ylabel(r'$\mathbf{\Delta W}$', fontsize=16)               
        plt.ylim(set_ylim)        
        plt.title('W = ' + str(wref), fontsize=20)
        plt.tight_layout(pad=1.0)
        plt.legend() #(fontsize=LEGEND_FONT)


#%%############################################################################
#                       Save model    
###############################################################################

if args.save_data: 
    
    if args.num_weights_mode == 0: 
        torch.save(final_model.state_dict(), "final_model_N"+str(args.hidden_size)+"NE"+str(args.num_epoch)\
                           +"NLD"+str(args.num_dnnrnn_layers)+"END"+str(args.END)+".pth") 
    else:                
        torch.save(final_model.state_dict(), "final_model_N"+str(args.hidden_size)+"NE"+str(args.num_epoch)\
                           +"NLD"+str(args.num_dnnrnn_layers)+"WM"+str(args.num_weights_mode)+"END"+str(args.END)+".pth") 
        
        
