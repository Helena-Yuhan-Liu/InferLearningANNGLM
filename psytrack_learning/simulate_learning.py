import numpy as np
from scipy.special import expit


def reward_max(W, X, y, r, side, i, base=None):
    pR = expit(X[i-1] @ W[i-1])
    return pR * (1-pR) * X[i-1] * (-1)**(side[i-1]+1) / (1/2)


def predict_max(W, X, y, r, side, i, base=None): 
    pCorrect = np.abs((1-side[i-1]) - expit(X[i-1] @ W[i-1]))
    return (1-pCorrect) * X[i-1] * (-1)**(side[i-1]+1) / 1


def reinforce(W, X, y, r, side, i, base=None):
    pChoice = np.abs((1-y[i-1]) - expit(X[i-1] @ W[i-1]))
    return (1-pChoice) * X[i-1] * (-1)**(y[i-1]+1) * r[i-1]


def reinforce_base(W, X, y, r, side, i, base):
    pChoice = np.abs((1-y[i-1]) - expit(X[i-1] @ W[i-1]))
    return (1-pChoice) * X[i-1] * (-1)**(y[i-1]+1) * (r[i-1] - base)


def simulate_learning(X, side, sigma, alpha, learning_rule, 
                      base=0, sigma0=0, W0=0, seed=None):
    '''Simulates weights, choices, and rewards for a given task and learning rule. 
    '''
    
    N, K = X.shape

    # Can calculate the noise added to each weight on each trial in advance
    rng = np.random.default_rng() # different noise for different animals to study -ve baseline 
    noise = rng.normal(scale=sigma, size=(N, K)) # np.random.normal
    noise[0] = rng.normal(scale=sigma0, size=K) # np.random.normal
    np.random.seed(seed) # put seed back in 
    noise = np.zeros((N,K))  # remove noise, comment out for the noisy experiments! 

    # Inputs
    W = np.zeros((N,K))  # weights
    y = np.zeros(N)      # choice {0,1}
    y_ = np.copy(y)
    r = np.zeros(N)      # reward {0,1}

    # Initialize weights, choice, and reward on first trial
    W[0] = noise[0] + W0
    y[0] = (np.random.rand() < expit(X[0] @ W[0])).astype(int)
    r[0] = (y[0]==side[0]).astype(int)

    # Iterate through remaining N-1 trials
    for i in range(1,N): 

        # Calculate the learning update from the last trial
        learning_update = learning_rule(W, X, y, r, side, i, base)

        # Update the weights
        W[i] = W[i-1] + noise[i] + alpha*learning_update 

        # Calculate choice on current trial
        y_[i] = expit(X[i] @ W[i])
        y[i] = (np.random.rand() < y_[i]).astype(int)
        # Calculate reward
        r[i] = (y[i]==side[i]).astype(int)

    return W.T, y, y_, r, noise