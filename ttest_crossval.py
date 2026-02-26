# -*- coding: utf-8 -*-
"""
Significance tests on cross-validation results  

@author: hyliu
"""

import numpy as np 
import scipy.stats as stats 

# # worse
modelA_folder = './saved_pdf/seeds_2weights/' # reinforce_base: 2 weights stim+bias 

# better  
modelB_folder = './saved_pdf/DNNval_seeds/' # DNNGLM 
#modelB_folder = './saved_pdf/RNNval_seeds/' # RNNGLM 

all_animal_list = ['CSHL_001', 'CSHL_002', 'CSHL_003', 'CSHL_004', 'CSHL_005', 'CSHL_006', 'CSHL_007',
                    'CSHL_008', 'CSHL_010', 'CSHL_012', 'CSHL_014', 'CSHL_015']
seed_list = [99, 41, 42, 43]

val_LL_A_list_seeds = [] # stores mean LL for each seed 
val_LL_B_list_seeds = []
for sd in seed_list: 
    val_LL_A_seed = []; val_LL_B_seed = []
    for animal_tt in all_animal_list: 
        val_LL_A = np.load(modelA_folder + animal_tt+'_seed' + str(sd)+'_valLL.npy') 
        val_LL_B = np.load(modelB_folder + animal_tt+'_seed' + str(sd)+'_valLL.npy')         
        val_LL_A_seed.append(val_LL_A); val_LL_B_seed.append(val_LL_B) 
    val_LL_A_list_seeds.append(np.mean(np.array(val_LL_A_seed)))
    val_LL_B_list_seeds.append(np.mean(np.array(val_LL_B_seed))) 
    
val_LL_A_list_animals = [] # stores mean LL for each animal
val_LL_B_list_animals = []
for animal_tt in all_animal_list: 
    val_LL_A_animal = []; val_LL_B_animal = []
    for sd in seed_list: 
        val_LL_A = np.load(modelA_folder + animal_tt+'_seed' + str(sd)+'_valLL.npy') 
        val_LL_B = np.load(modelB_folder + animal_tt+'_seed' + str(sd)+'_valLL.npy')         
        val_LL_A_animal.append(val_LL_A); val_LL_B_animal.append(val_LL_B) 
    val_LL_A_list_animals.append(np.mean(np.array(val_LL_A_animal)))
    val_LL_B_list_animals.append(np.mean(np.array(val_LL_B_animal))) 
     
val_LL_A_list_seeds = np.array(val_LL_A_list_seeds)
val_LL_B_list_seeds = np.array(val_LL_B_list_seeds)

# print mean & std 
print(f"modelA Per-trial LL: {np.mean(val_LL_A_list_seeds):.4f} +/- {np.std(val_LL_A_list_seeds):.4f}")
print(f"modelB Per-trial LL: {np.mean(val_LL_B_list_seeds):.4f} +/- {np.std(val_LL_B_list_seeds):.4f}")

# # Perform a paired t-test: where each entry corresponds to the same animal  
t_stat, p_value = stats.ttest_rel(np.array(val_LL_A_list_animals), np.array(val_LL_B_list_animals), alternative='less')
print(f"P-value (animals): {p_value:.2e}")
