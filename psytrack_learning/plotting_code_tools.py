# -*- coding: utf-8 -*-
import os
import re
import logging
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from scipy.optimize import nnls
from scipy.optimize import curve_fit
from scipy.special import expit as sigmoid  
from scipy.interpolate import griddata 
from scipy.stats import linregress
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import random 
# from brokenaxes import brokenaxes 

import psytrack_learning as psy    
from psytrack_learning.simulate_learning import simulate_learning
colors = psy.COLORS
from models import DeltaDNNGLM 

# Set matplotlib defaults from making files editable and consistent in Illustrator 
colors = psy.COLORS
zorder = psy.ZORDER
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = (1,1,1,0)
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = 10
# plt.rcParams['font.family'] = 'cmu serif'
plt.rcParams['font.family'] = 'Arial' #'Helvetica'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['axes.labelsize'] = 13

# Set the logging level to suppress warnings
logging.getLogger('matplotlib').setLevel(logging.ERROR)
# Alternatively, you can disable font logging specifically
matplotlib.font_manager._log.setLevel(logging.ERROR)

TITLE_FONT = 16 
AXIS_FONT = 16
LEGEND_FONT = 14 

def remove_box(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return ax

#%%############################################################################
#                       Helper functions  
###############################################################################

def set_fonts_global():
    return TITLE_FONT, AXIS_FONT, LEGEND_FONT 

def aggregate_unique_x(x, y):
    unique_x, indices = np.unique(x, return_inverse=True)
    mean_y = np.zeros_like(unique_x, dtype=float)
    std_y = np.zeros_like(unique_x, dtype=float)
    
    for i, ux in enumerate(unique_x):
        mask = (x == ux)
        mean_y[i] = np.mean(y[mask])
        std_y[i] = np.std(y[mask], ddof=1)  # Using ddof=1 for sample std deviation
    
    return unique_x, mean_y, std_y

def reinforce_Warray(W, X, y, r, answer, Wbias):
    pChoice = np.abs((1-y) - expit(X*W + Wbias))
    return (1-pChoice) * X * (-1)**(y+1) * r

def plot_interpolated_heatmap(x, w, dw, plt, set_ylim, wref_list, color_list=None):
    # Create a structured grid
    grid_x, grid_w = np.linspace(min(x), max(x), 50), np.linspace(min(w), max(w), 50)
    grid_X, grid_W = np.meshgrid(grid_x, grid_w)

    # Interpolate dw onto the grid
    grid_dw = griddata((x, w), dw, (grid_X, grid_W), method='cubic')

    # Plot heatmap
    plt.imshow(grid_dw, extent=[x.min(), x.max(), w.min(), w.max()], 
               origin='lower', cmap='viridis', aspect='auto', vmin=0.5*set_ylim[0], vmax=0.5*set_ylim[1])
    #plt.colorbar(label="Interpolated dw values")
    # Add horizontal lines at wref values
    for ww in range(len(wref_list)):
        if color_list is None: 
            plt.axhline(y=wref_list[ww], color='white', linestyle='--', linewidth=2.0, alpha=0.8)    
        else:
            plt.axhline(y=wref_list[ww], color=color_list[ww], linestyle='--', linewidth=2.0, alpha=0.8)  
    plt.xlabel(r'Stimulus $s$', fontsize=AXIS_FONT)
    plt.ylabel(r'$Wstim$', fontsize=AXIS_FONT) 
    

#%%############################################################################
#                       Plotting code   
###############################################################################

def plot_grid(fsave_prefix, set_ylim, set_yticks=None, all_on_one = True, small_plot=False): 
    
    test_data = torch.load(fsave_prefix + "test_data.pth")   
    test_inputs = test_data['test_inputs'] 
    args = torch.load(fsave_prefix + "args.pth")
    sim_learning_rule = args['learning_rule_name'] 
    alpha = 2**(args['alpha_pow'])
    
    input_size = test_inputs.shape[-1]+1
    final_model = DeltaDNNGLM(input_size, args['hidden_size'], args['trunc_len'], args['num_layers'])  

    final_model.load_state_dict(torch.load(fsave_prefix + "final_model.pth")) 
    final_model.eval() 
    
    num_wvals = 1001         
    eps = 0.01
    Mwindow = 1 
    
    # Setup 
    x_val = np.array([-2., -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0., 
                      0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.]) 
    y_val = np.array([0., 1.]) 
    w_val = np.linspace(-2, 2, num_wvals) 
    b_val = np.array([-1.0, 0., 1.0]) 

    # Create the grid and flatten it
    X_, Y_, W_, B_ = np.meshgrid(x_val, y_val, w_val, b_val, indexing='ij')
    stimulus = X_.ravel()
    choices = Y_.ravel()
    w_vals = W_.ravel()
    b_vals = B_.ravel() 

    # Convert to torch tensors in float type
    stimulus = torch.tensor(stimulus, dtype=torch.float32).unsqueeze(-1)
    choices = torch.tensor(choices, dtype=torch.float32).unsqueeze(-1)
    w_vals = torch.tensor(w_vals, dtype=torch.float32).unsqueeze(-1)
    b_vals = torch.tensor(b_vals, dtype=torch.float32).unsqueeze(-1)
    w_stack = torch.cat([w_vals, b_vals], axis=-1)

    # Define answer and reward_data based on conditions
    answer_data = (stimulus > 0).float()
    reward_data = (choices == answer_data).float()
    dnn_inputs = torch.cat([w_stack, stimulus, torch.ones_like(stimulus), choices, reward_data, answer_data], dim=-1)

    # Get dW and plot     
    dWdnn = final_model.dnn(dnn_inputs)*final_model.scaling_factor 
    delta_Wrec_stack = dWdnn[:,0:1].detach().numpy()
    # delta_brec_stack = dWdnn[:,1].detach().numpy() 
    Wrec_stack = w_vals.detach().numpy()
    brec_stack = b_vals.detach().numpy()

    # Plot + fit 
    xx_ = stimulus.detach().numpy()
    yy_ = choices.detach().numpy()
    ww_ = w_vals.detach().numpy()
    bb_ = b_vals.detach().numpy() 
    zz_ = answer_data.detach().numpy() 
    rr_ = reward_data.detach().numpy()
        
    if sim_learning_rule == 'reinforce': 
        deltaW_actual = alpha*reinforce_Warray(ww_, xx_, yy_, rr_, zz_, bb_)    
        
    rmse = np.sqrt(np.mean((deltaW_actual - delta_Wrec_stack) ** 2))
    print(f'RMSE: {rmse:.4f}') 
        
    if all_on_one: 
        if small_plot:     
            fig_dWstack_heat_wslices = plt.figure(figsize=(8, 4))
        else:
            fig_dWstack_heat_wslices = plt.figure(figsize=(12, 8))
    else: 
        if small_plot: 
            fig_dWstack_heat_wslices = plt.figure(figsize=(12, 4))
        else:
            fig_dWstack_heat_wslices = plt.figure(figsize=(12, 12)) 
    remove_box()
    plot_ii = 0
    if small_plot: 
        bref_list = [0]
    else:
        bref_list = [-1, 0, 1] 
    y_smooth_list = [] 
    for bref in bref_list: #[0]: 
        
        def plot_wslices(wref, deltaW, color, plt, correct=True): 
            if correct:
                mask = (np.abs(ww_ - wref) < eps) & ((yy_ > 0.5) == (xx_ > 0.)) & (np.abs(bb_ - bref) < eps) 
            else: 
                mask = (np.abs(ww_ - wref) < eps) & ((yy_ > 0.5) != (xx_ > 0.)) & (np.abs(bb_ - bref) < eps) 
            # x_smooth, y_smooth, _, _ = sort_and_smooth(xx_[mask], deltaW[mask], M=Mwindow) 
            x_smooth, y_smooth, _ = aggregate_unique_x(xx_[mask], deltaW[mask])
            if all_on_one:
                wlabel = f"W={wref}{', correct' if correct else ', incorrect'}"
            else:
                wlabel = 'W='+str(wref)
            if correct:
                plt.plot(x_smooth, y_smooth, label=wlabel, color=color, linestyle='-', linewidth=2.0) 
            else: 
                if sim_learning_rule == 'predict_max': 
                    y_smooth = y_smooth + 1e-3 # to make incorrect visible 
                plt.plot(x_smooth, y_smooth, label=wlabel, color=color, linestyle='--', linewidth=2.0) 
            plt.xlabel(r'Stimulus $s$', fontsize=AXIS_FONT)
            plt.ylabel(r'$\Delta Wstim$', fontsize=AXIS_FONT)               
            plt.ylim(set_ylim)  
            if set_yticks is None:
                mid = (set_ylim[0] + set_ylim[1]) / 2
                plt.yticks([set_ylim[0], mid, set_ylim[1]])
            else:
                plt.yticks(set_yticks)
            if bref == bref_list[0]:
                pass #plt.legend(fontsize=LEGEND_FONT)
            else:
                ax.set_yticklabels([])
                ax.set_yticks([])
            return x_smooth, y_smooth 
           
        wref_list = [-1, 0, 1] #[0, 1] # [-1, 0, 1] 
        wcolor_list = ['m', 'g', 'b'] #['m', 'r'] # ['m', 'g', 'b']
        if all_on_one: 
            # sim     
            if small_plot: 
                ax = plt.subplot(121) 
            else:
                ax = plt.subplot(2, len(bref_list), plot_ii+1) 
            remove_box(ax)
            for wref, wcolor in zip(wref_list, wcolor_list):
                x_smooth, y_smooth = plot_wslices(wref, deltaW_actual, wcolor, plt, correct=True) 
                x_smooth, y_smooth = plot_wslices(wref, deltaW_actual, wcolor, plt, correct=False)
            if small_plot:
                plt.title('true', fontsize=TITLE_FONT)
            else:
                plt.title('Wbias = ' + str(bref) + '\ntrue', fontsize=TITLE_FONT) 
            # ax.set_xticklabels([])
            # ax.set_xticks([])
            
            # fit             
            if small_plot: 
                ax = plt.subplot(122) 
            else:
                ax = plt.subplot(2, len(bref_list), plot_ii+1 + len(bref_list)) 
            remove_box(ax)
            for wref, wcolor in zip(wref_list, wcolor_list):
                x_smooth, y_smooth = plot_wslices(wref, delta_Wrec_stack, wcolor, plt, correct=True) 
                y_smooth_list.append(y_smooth[None,:]) 
                x_smooth, y_smooth = plot_wslices(wref, delta_Wrec_stack, wcolor, plt, correct=False) 
                y_smooth_list.append(y_smooth[None,:])   
            plt.title('fitted', fontsize=TITLE_FONT) 
            
        else:   
            # sim heat, correct  
            mask = ((yy_ > 0.5) == (xx_ > 0.)) & (np.abs(bb_ - bref) < eps) 
            if small_plot: 
                ax = plt.subplot(131) 
            else:
                ax = plt.subplot(3, len(bref_list), plot_ii+1) 
            remove_box(ax)
            plot_interpolated_heatmap(xx_[mask], ww_[mask], deltaW_actual[mask], plt, set_ylim, wref_list, wcolor_list) 
            if small_plot: 
                plt.title('true, after correct y', fontsize=TITLE_FONT)
            else:
                plt.title('Wbias = ' + str(bref), fontsize=TITLE_FONT) 
            # fig_dWstack_heat_wslices.suptitle("Ground truth ∆W after correct choice", fontsize=TITLE_FONT) 
            # sim         
            if small_plot: 
                ax = plt.subplot(132) 
            else:
                ax = plt.subplot(3, len(bref_list), plot_ii+1 + len(bref_list)) 
            remove_box(ax)
            for wref, wcolor in zip(wref_list, wcolor_list):
                x_smooth, y_smooth = plot_wslices(wref, deltaW_actual, wcolor, plt, correct=True) 
                x_smooth, y_smooth = plot_wslices(wref, deltaW_actual, wcolor, plt, correct=False)
            plt.title('true', fontsize=TITLE_FONT) 
            # ax.set_xticklabels([]) 
            # ax.set_xticks([])
            # fit     
            if small_plot: 
                ax = plt.subplot(133) 
            else: 
                ax = plt.subplot(3, len(bref_list), plot_ii+1 + 2*len(bref_list)) 
            remove_box(ax)
            for wref, wcolor in zip(wref_list, wcolor_list):
                x_smooth, y_smooth = plot_wslices(wref, delta_Wrec_stack, wcolor, plt, correct=True)
                x_smooth, y_smooth = plot_wslices(wref, delta_Wrec_stack, wcolor, plt, correct=False) 
            plt.title('fitted', fontsize=TITLE_FONT)                 
        
        plot_ii += 1        
    plt.tick_params(axis='both', which='major', labelsize=12)  # Set font size for axis ticks
    plt.tight_layout(pad=3.0)
    plt.show() 
    
    if small_plot:
        fig_dWstack_heat_wslices.savefig(fsave_prefix + 'fig_dWstack_heat_wslices_smallplot.pdf')
    else:
        fig_dWstack_heat_wslices.savefig(fsave_prefix + 'fig_dWstack_heat_wslices.pdf')
     
    return rmse