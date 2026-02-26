# -*- coding: utf-8 -*-
"""
Run REINFORCE for the IBL dataset 

@author: hyliu
"""

import os
import re
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
import numpy as np
import pandas as pd
import torch
import random  

import psytrack_learning as psy
from psytrack_learning.helper.helperFunctions import update_hyper, hyper_to_list
from psytrack_learning.simulate_learning import reward_max, predict_max, reinforce, reinforce_base
# from psytrack_learning.simulate_learning import simulate_learning
from psytrack_learning.hyperparameter_optimization import sigmoid, BCEloss, det_logLL_animals

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--fold', default=0, type=int, help='Fold index (0-4, assuming 5-fold cross-val)')
parser.add_argument('--seed', default=99, type=int, help='seed (the paper uses 99, 41, 42, 43')
parser.add_argument('--END', default=10000, type=int, help='data END')
parser.add_argument('--learning_rule_name', default='reinforce_base', type=str, help='Learning rule simulated: reinforce_base, etc.')
args = parser.parse_args()

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

# Set save path for all figures
spath = "./Figures/"  # UPDATE

FOLDS=5
fold=args.fold
assert (fold>=0)and(fold<=FOLDS), "fold index must be within the appropriate range"

deterministic_learning = True 
assert deterministic_learning, "to keep the comparison within the same setting, focus on the deterministic part of the learning" 
rec_learning_rule = globals()[args.learning_rule_name] 

#%%############################################################################
#                       Load/process data 
############################################################################### 

# Mouse data 
END = args.END 
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

def list_to_hyper(x, K):
    """
    Convert a 1D list back into the hyper dictionary, in the same order.
    """
    # alpha occupies the first K entries
    alpha = x[0 : K]
    # adder occupies the next K
    adder = x[K : 2*K]
    
    hyper = {
        'alpha': alpha,
        'adder': adder,
    }
    return hyper

# try constraining baseline to be positive? 
def constraint_adder(params):
    hyper_dict = list_to_hyper(params, K)  # Convert list back to dict
    adder_values = hyper_dict['adder']  # Extract adder list

    # Ensure both entries in 'adder' are non-negative
    print(f"Constraint check: adder = {adder_values}")  # Debugging output
    return min(adder_values)  # Ensures the smallest value is ≥ 0

all_animal_list = ['CSHL_001', 'CSHL_002', 'CSHL_003', 'CSHL_004', 'CSHL_005', 'CSHL_006', 'CSHL_007',
                    'CSHL_008', 'CSHL_010', 'CSHL_012', 'CSHL_014', 'CSHL_015']

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

indices = np.random.permutation(len(all_animal_list))
test_indices = np.array_split(indices, FOLDS)
test_indices = [list(test_set) for test_set in test_indices]

test_animal_list = [all_animal_list[i] for i in test_indices[fold]]
train_animal_list = [all_animal_list[i] for i in indices if i not in test_indices[fold]] 
# train_animal_list = all_animal_list # debugging, to get train accuracy 
# test_animal_list = all_animal_list 


#%%############################################################################
#                       Train 
###############################################################################

weights = {'bias' : 1, 'cL' : 0, 'cR' : 0, 'cBoth': 1} # The two weights version
K = np.sum([weights[i] for i in weights.keys()])

alpha_guess = [2**-3, 2**-8]   
        
hyper_guess = {
    'alpha': alpha_guess, # [2**-6] * K
    'adder': [0.] * K, # [0.] * K 
    'W0': [0.] * K, 
}   

optList = ['alpha', 'adder']
loss2min = det_logLL_animals  

# Collect data from animals 
dat_list = []

for animal_ii in train_animal_list:
    dat = psy.trim(getMouse(animal_ii, 5), START=0, END=END)
    dat['inputs']['h'] = np.concatenate(([0.], dat['inputs']['cBoth'][:-1,0]))[:,None] 
    X = psy.read_input(dat, weights)
    dat['X'] = X 
    dat_list.append(dat)

# List of extra arguments used by evd_lossfun in optimization of evidence
options = {'maxiter': 500, 'tol':1e-4}
args = {"optList": optList, "dat_list": dat_list, "K": K, "learning_rule": rec_learning_rule,
        "hyper": hyper_guess, "weights": weights, "update_w": True, "wMode_list": [None] * len(train_animal_list),
        "tol": 1e-6, "showOpt": True}

# # Default
res = minimize(loss2min, hyper_to_list(hyper_guess, optList, K), args=args, method='COBYLA', options=options)
# # With constraints 
# constraints = [{'type': 'ineq', 'fun': constraint_adder}]
# res = minimize(loss2min, hyper_to_list(hyper_guess, optList, K), args=args, method='SLSQP', constraints=constraints, options=options)
print("Evidence:", -res.fun, "  ", optList, ": ", res.x)

opt_hyper = update_hyper(res.x, optList, hyper_guess, K)
print('opt_hyper: ')
print(opt_hyper)

#%%############################################################################
#                       Test on data 
###############################################################################

alpha_fit = opt_hyper['alpha']
adder_fit = opt_hyper['adder']
W0_fit = opt_hyper['W0']

log_likelihood_val_list = []
val_accuracy_list = []
sim_colors = [colors['bias'], colors['cBoth']] 

for animal_tt in test_animal_list:
    dat = psy.trim(getMouse(animal_tt, 5), START=0, END=END) 
    # dat['inputs']['h'] = np.concatenate(([0.], dat['inputs']['cBoth'][:-1,0]))[:,None]
    
    X = psy.read_input(dat, weights)
    y = dat['y']
    answer = dat['answer'] 
    r = (y==answer).astype(int) 
    wMode = np.zeros((len(y),K))
    wMode[0,:] = W0_fit 
    for tt in range(1,len(y)):
        learning_update = rec_learning_rule(wMode, X, y, r, answer, tt, adder_fit) 
        wMode[tt] = wMode[tt-1] + alpha_fit*learning_update # deterministic learning component 
    wMode = wMode.T 
    
    # Get LL & accuracy 
    py_pred = sigmoid(np.sum(wMode.T*X, axis=1)) 
    log_likelihood_val = -BCEloss(y, py_pred)
    val_accuracy = np.sum((py_pred > 0.5) == y) / len(y)
    log_likelihood_val_list.append(log_likelihood_val)
    val_accuracy_list.append(val_accuracy)
    ## For total loglikelihodd, multiply the total number of trials (10000) and animals (12)
    print('Per-trial Log-Likelihood of ' + animal_tt + f': {log_likelihood_val:.4f}') 
    save_dir = './saved_pdf/seeds_2weights'
    os.makedirs(save_dir, exist_ok=True)    
    np.save(f'{save_dir}/{animal_tt}_seed{seed}_valLL.npy', log_likelihood_val)
    
    # # Uncomment to plot and save weights 
    # fig_val = plt.figure(figsize=(3.25,1.25))
    # for i, c in enumerate(sim_colors):
    #     plt.plot(wMode[i], c=c, lw=1, linestyle='-', alpha=0.85, zorder=2*i+1)
    # plt.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)
    # plt.xticks(1000*np.arange(0,7))
    # plt.yticks(np.arange(-2,3,2))
    # plt.xlim(0,END)
    # #plt.ylim(-3.5,3.5)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.show()
    # fsave_prefix = './saved_pdf/seeds_2weights/psytrack_animal_' + animal_tt + '_seed' + str(seed) 
    # fig_val.savefig(fsave_prefix + '_valWrec.pdf')
    # np.save(fsave_prefix + '_PsyLearnWval.npy', wMode)
    
log_likelihood_val_mean = np.mean(np.array(log_likelihood_val_list))
val_accuracy_mean = np.mean(np.array(val_accuracy_list))
print(f'Per-trial Log-Likelihood: {log_likelihood_val_mean:.4f}')
print(f'Accuracy: {val_accuracy_mean * 100:.2f}%') 
    
