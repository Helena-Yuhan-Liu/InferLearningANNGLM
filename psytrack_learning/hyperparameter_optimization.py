import numpy as np

from .getMAP import getMAP
from .helper.helperFunctions import update_hyper

# Optimize LL instead based on deterministic weight trajectory 
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def BCEloss(y_true, y_pred):
    # Ensure predictions are within the valid range (avoid log(0))
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute the BCE loss for each sample
    bce_loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Return the average loss over the batch
    return np.mean(bce_loss)

def det_logLL_animals(vals, keywords):

    # Update the hyper dict with the current guesses
    hyper = update_hyper(vals, keywords['optList'], keywords['hyper'], keywords['K'])
    lr = keywords['learning_rule']
    # lr = lr if lr is None else globals()[lr]
    
    logLL_total = 0 
    alpha_fit = hyper['alpha']
    adder_fit = hyper['adder']
    W0_fit = hyper['W0']
    for ii in range(len(keywords['dat_list'])):
        # Recover the weights and evidence for the current hyper guess
        dat = keywords['dat_list'][ii]
        X = dat['X'] 
        y = dat['y']
        K = keywords['K'] 
        wMode = np.zeros((len(y),K))
        wMode[0,:] = W0_fit 
        answer = dat['answer']
        r = (y==answer).astype(int) 
        for tt in range(1,len(y)):
            learning_update = lr(wMode, X, y, r, answer, tt, adder_fit) 
            wMode[tt] = wMode[tt-1] + alpha_fit*learning_update # deterministic learning component 
        wMode = wMode.T   
        
        # get logLL 
        py_pred = sigmoid(np.sum(wMode.T*X, axis=1)) 
        logLL = -BCEloss(y, py_pred)

        # When optimizing hypers, update the initial guess of weights for the next iteration
        if keywords["update_w"]:
            keywords["wMode_list"][ii] = wMode
            if "iter" in keywords:
                keywords["iter"] += 1
            else:
                keywords["iter"] = 1
        else:
            keywords["iter"] = 0
            
        logLL_total += logLL / len(keywords['dat_list'])
        
    # Print fitting info if desired
    if "showOpt" in keywords and keywords["showOpt"]:
        print("   ", np.round(vals,3), keywords["iter"], np.round(-logLL_total,3),
              "")
        
    return -logLL_total 

# def evd_lossfun(vals, keywords):

#     # Update the hyper dict with the current guesses
#     hyper = update_hyper(vals, keywords['optList'], keywords['hyper'], keywords['K'])
#     lr = keywords['learning_rule']
#     # lr = lr if lr is None else globals()[lr]
    
#     # Recover the weights and evidence for the current hyper guess
#     wMode, _, logEvd, _ = getMAP(
#         keywords['dat'],
#         hyper,
#         keywords['weights'],
#         W0=keywords["wMode"],
#         learning_rule=lr,
#         tol=keywords["tol"],
#         showOpt=0)

#     # When optimizing hypers, update the initial guess of weights for the next iteration
#     if keywords["update_w"]:
#         keywords["wMode"] = wMode
#         if "iter" in keywords:
#             keywords["iter"] += 1
#         else:
#             keywords["iter"] = 1
#     else:
#         keywords["iter"] = 0
    
#     # Print fitting info if desired
#     if "showOpt" in keywords and keywords["showOpt"]:
#         print("   ", np.round(vals,3), keywords["iter"], np.round(-logEvd,3),
#               "")
        
#     return -logEvd

# def evd_lossfun_animals(vals, keywords):

#     # Update the hyper dict with the current guesses
#     hyper = update_hyper(vals, keywords['optList'], keywords['hyper'], keywords['K'])
#     lr = keywords['learning_rule']
#     # lr = lr if lr is None else globals()[lr]
    
#     logEvd_total = 0
#     for ii in range(len(keywords['dat_list'])):
#         # Recover the weights and evidence for the current hyper guess
#         wMode, _, logEvd, _ = getMAP(
#             keywords['dat_list'][ii],
#             hyper,
#             keywords['weights'],
#             W0=keywords["wMode_list"][ii],
#             learning_rule=lr,
#             tol=keywords["tol"],
#             showOpt=0)

#         # When optimizing hypers, update the initial guess of weights for the next iteration
#         if keywords["update_w"]:
#             keywords["wMode_list"][ii] = wMode
#             if "iter" in keywords:
#                 keywords["iter"] += 1
#             else:
#                 keywords["iter"] = 1
#         else:
#             keywords["iter"] = 0
            
#         logEvd_total += logEvd 
        
        
#     # Print fitting info if desired
#     if "showOpt" in keywords and keywords["showOpt"]:
#         print("   ", np.round(vals,3), keywords["iter"], np.round(-logEvd_total,3),
#               "")
        
#     return -logEvd_total 
