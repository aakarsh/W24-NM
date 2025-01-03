#%%

from  datetime import datetime
from joblib import Parallel, delayed
from numba import njit
from scipy.optimize import minimize
import jax
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
import random
import seaborn as sns

sns.set_theme(style='white', context='notebook', font_scale=1.2)
#%%
df = pd.read_csv('gen_data.csv')

# Go+ = Go to win
# Go- = Go to avoid losing
# NoGo+ = Don't go to win
# NoGo- = Don't go to avoid losing
cue_mapping = { 1: 'Go+', 2: 'Go-', 3: 'NoGo+', 4: 'NoGo-' }  
#%%
"""
\subsection{Task-1:}

* It contains the data of 10 subjects :
    (see column ”ID” for the subject identifier)
     performing a go/no-go task, each for 600 trials. 
     
    [0-9] ID: subject identifier
    600 - Number of trials
    
*  The column ”cue” informs you about the presented trial type 
    (see the ”cue mapping” variable in our template). 

    (1) - Go+
    (2) - Go-
    (3) - NoGo+
    (4) - NoGo-
    
* "pressed" contains the response of the participant 
    (0) - no-go
    (1) - go
    
* "outcome" contains whether  :
    (1)  - reward was delivered  
    (0)  - nothing was delivered 
    (-1) -  a punishment was given 
"""
#%%
# \section{Exercise-1}: Plot the Accuracy for each Cue
#
"""
    Recreate figure 2E of the paper :
    "
    Go and no-go learning in reward and punishment: 
        Interactions between affect and effect” 
        
    Only the bar plots are important here, no need for error 
    bars or significance tests.
"""
def compute_accuracy(df, cue_mapping=cue_mapping):
    accuracy_map = {}
    for cue in cue_mapping:
        # Subset the data for the current cue
        cue_df = df[df['cue'] == cue]
        if cue_mapping[cue] == 'Go+': 
            # Cue type was Go+ (That is go to win)
            # Subject answers presses the button to win reward
            # Thus accuracy is the mean of the pressed column
            accuracy_map[cue_mapping[cue]] = cue_df['pressed'].mean()
            print(f'Go+: {accuracy_map[cue_mapping[cue]]}')
        elif cue_mapping[cue] == 'Go-':
            # Cue type was Go- : Go to avoid punishment
            # Subject answers presses the button to 
            # avoid punishment 
            # Thus accuracy is the mean of the pressed column
            accuracy_map[cue_mapping[cue]] = cue_df['pressed'].mean()
            print(f'Go-: {accuracy_map[cue_mapping[cue]]}')
        elif cue_mapping[cue] == 'NoGo+':
            # Subject avoids pressing the button, to win reward
            # Thus accuracy is the inverse of the mean of the pressed column 
            accuracy_map[cue_mapping[cue]] = 1 - cue_df['pressed'].mean()
            print(f'NoGo+: {accuracy_map[cue_mapping[cue]]}')
        elif cue_mapping[cue] == 'NoGo-':
            # Thus accuracy is the inverse of the mean of the pressed column 
            accuracy_map[cue_mapping[cue]] = 1 - cue_df['pressed'].mean()
            print(f'NoGo-: {accuracy_map[cue_mapping[cue]]}')
    return accuracy_map
    
def plot_cue_accuracy(df, save_path='cue_accuracy.png'):
    accuracy_map = compute_accuracy(df)
    # Plot the accuracy as a bar plot, with seaborn
    labels = {
        'Go+': 'Go to Win',
        'Go-': 'Go to avoid Losing',
        'NoGo+': "Don't go to Win :(",
        'NoGo-': "Don't go to avoid Losing"
    }

    label_list = [labels[k] for k in accuracy_map.keys()]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=label_list, y=list(accuracy_map.values()))
    plt.ylabel('Accuracy')
    plt.title('Accuracy for each Cue')
    plt.savefig(save_path)
    plt.show()

## Plot the accuracy for each cue. 
# Note that Don't go to win has lowest accuracy, while 
# Go to Win has highest accuracy.
plot_cue_accuracy(df)
accuracy_map = compute_accuracy(df)
assert accuracy_map['Go+'] >= accuracy_map['NoGo+']
assert min(accuracy_map.values()) == accuracy_map['NoGo+']
assert max(accuracy_map.values()) == accuracy_map['Go+']
#%%
"""
\subsection{Task-2}:

Program the log-likelihood functions of the models 1 to 7.
(including) presented in 
    "Disentangling the Roles of Approach, Activation and 
    Valence in Instrumental and Pavlovian Responding" 
    
    (see Table 2 of that paper for the model numbering and 
    relevant parameters). 
    
    The paper uses these parameters:
            - Learning Rate ε:
            - Feedback Sensitivity β:
                - The general feedback sensitivity β. 
                    Can be replaced by separate reward and punishment 
                    sensitivities ρ 
                    (We don't include a sensitivity for omission) 
                    - There can be different learning rates ε 
                for:
                    - Reward: 
                    - Feedback Omission/No-reward-No-punishment 
                    - Punishment 
                        (The paper doesn't make use of omissions, 
                        so they use only two learning rates, 
                        you will need three).
                    - There can be a: 
                        - $bias_{app}$  - General bias to approach. 
                        - $bias_{wth}$  - General bias to withhold responding.
"""
# Define yourself a softmax function
@njit
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0) 

def subject_df(df, subject_id):
    return df[df['ID'] == subject_id]

def num_cues(subject_df):
    return len(np.unique(subject_df['cue']))

def num_states(subject_df):
    return num_cues(subject_df) 

def num_trials(subject_df):
    return len(subject_df)

def num_actions(subject_df):
    return len([1, 0]) 
@njit
def num_outcomes(subject_df):
    return len([1, 0, -1])

@njit
def empty_q(num_cues, num_actions):
    return np.zeros((num_cues, num_actions))

#%%
first_subject_df = subject_df(df, 0)

#%%
def model_negative_log_likelihood(data, 
                                  q_update, 
                                  q_init=empty_q, 
                                  q_bias=None):
    """
        data: pd.DataFrame
            The data of one subject
        q_update: function
            The function to update the Q-values
        model_params: dict
            The parameters of the model
    """
    # Run a q-learning model on the data,  return the 
    # log-likelihood parameters are learning rate and beta, 
    # the feedback sensitivity.
    q = q_init(num_states(data), num_actions(data))
    
    log_likelihood = 0

    for t in range(len(data)):
        # Extract the current trial
        trial = data.iloc[t]
        # Extract the current state and action
        state = trial['cue'] - 1 # 0-indexed
        # Extract the subject's response
        action = trial['pressed']
        # Extract the reward
        reward = trial['outcome']
        
        # Compute the probability of the action
        
        prob = None
        if q_bias is None:
            prob = softmax(q[state])
        else:
            q_biased = q_bias(q, state) 
            prob = softmax(q_biased)
       
        # Compute the log-likelihood
        log_likelihood += np.log(prob[action])
        # Update the Q-values - Feedback sensitive prediction error.
        q[state, action] = q_update(q, state, action, reward)
        
    return -log_likelihood

# num-states: 4 (Go+, Go-, NoGo+, NoGo-)
# num-actions: 2 (1, 0)
# q[state] - 2 actions

#%%
"""
 Model-1 Assumes that:
    * \epsilon - learning rate
    * \beta - feedback sensitivity
        * $-\rho_{pun} = \rho_{rew} = \beta$
    * No bias parameters
        * bias_{app} = bias_{wth} = 0
"""
def model_1(data, learning_rate, beta):
    """
    data: pd.DataFrame
        The data of one subject
    learning_rate: float
        The learning rate parameter
    beta: float
        The feedback sensitivity parameter
    """
    @njit
    def update_rule(q, state, action, reward):
        """
        q: np.ndarray
            The Q-values
        state: int
            The current state
        action: int
            The current action
        reward: int
            The reward
        """
        prediction_error = (beta * reward) - q[state, action] 
        return q[state, action] + learning_rate * prediction_error
    
    return model_negative_log_likelihood(data, update_rule) 

#%%
"""
Model-2: Assumes that:
    * \epsilon - learning rate
    * \rho_pun ,\rho_rew - feedback sensitivity
        * \rho_pun != \rho_rew - (separate reward and punishment sensitivities) 
    * No bias parameters
        * bias_{app} = bias_{wth} = 0 
        
* Model Number: 2
* Parameters: \epsilon, \rho_{rew}, \rho_{pun}
* BIC: 4613
"""
def model_2(data, learning_rate, rho_rew, rho_pun):
    """
    data: pd.DataFrame
        The data of one subject
    learning_rate: float
        The learning rate parameter
    rho_rew: float
        The reward sensitivity parameter
    rho_pun: float
        The punishment sensitivity parameter
    """
    @njit
    def update_rule(q, state, action, reward):
        """
        q: np.ndarray
            The Q-values
        state: int
            The current state
        action: int
            The current action
        reward: int
            The reward
        """
        scaling_factor = (reward == 1) * rho_rew + (reward == -1) * rho_pun
        prediction_error = (scaling_factor * reward) - q[state, action]
        return q[state, action] + (learning_rate * prediction_error)
    return model_negative_log_likelihood(data, update_rule)

"""
Model-3: Assumes that:
    * \epsilon_new, \epsilon_rew - learning rates
    * \beta - feedback sensitivity
    * No bias parameters
        * bias_{app} = bias_{wth} = 0
        
Model Number: 3
Parameters: \epsilon_{rew}, \epsilon_{pun}, \epsilon_{omm}, \beta
Expected BIC: 4665
"""
def model_3(data, learning_rate_rew, learning_rate_pun, learning_rate_omm, beta):
    """
    """
    def update_rule(q, state, action, reward):
        """
        q: np.ndarray
            The Q-values
        state: int
            The current state
        action: int
            The current action
        reward: int
            The reward
        """
        # Create an array of learning rates corresponding to reward values
        learning_rates = np.array([learning_rate_pun, learning_rate_omm, learning_rate_rew])
        # Map reward values to indices: -1 -> 0, 0 -> 1, 1 -> 2
        reward_index = reward + 1
        # Select the appropriate learning rate
        learning_rate = learning_rates[reward_index]
        # Calculate the prediction error
        prediction_error = (beta * reward) - q[state, action]
        # Update the Q-value
        return q[state, action] + learning_rate * prediction_error
        
    return model_negative_log_likelihood(data, update_rule)  

"""
Model-4: Assumes that:

    * \epsilon - Learning Rate
    * \beta - Feedback Sensitivity
    * Biases: 
        * bias_{app} != bias_{wth} - (separate biases to approach and withhold responding)
        * bias_{app} - bias to approach  
        * bias_{wth} - bias to withhold responding
    b(a_t) - can take on 
        bias_{with} - for withdrawal actions 
        bias_{app} - for approach actions
        0 - for all nogo options
Model Number: 4
Parameters: \epsilon, \beta, bias_{app}, bias_{wth}
Expected BIC: 4771
"""
def model_4(data, learning_rate, beta, bias_app, bias_wth):
    """
    Model-4 includes biases.
    """
    def q_bias(q, state):   
        bias_matrix = np.array([
            [0.0, bias_app],  # Go+ (state 0): NoGo = 0, Go = bias_app
            [0.0, bias_wth],  # Go- (state 1): NoGo = 0, Go = bias_wth
            [0.0, 0.0],       # NoGo+ (state 2): No bias for either action
            [0.0, 0.0]        # NoGo- (state 3): No bias for either action
        ])
        return q[state] + bias_matrix[state] 
       
    def update_rule(q, state, action, reward):
        """
        q: np.ndarray
            The Q-values
        state: int
            The current state
        action: int
            The current action
        reward: int
            The reward
        """
        # Update Q-value with learning rate and prediction error
        prediction_error = (beta * reward) - q[state, action]
        return q[state, action] + (learning_rate * prediction_error)
    
    return model_negative_log_likelihood(data, update_rule, q_bias=q_bias)
"""
Model-5: Assumes that:
    * \epsilon - learning rate
    * \rho_pun ,\rho_rew - feedback sensitivity
    * bias_{app}, bias_{wth} - biases: 
        * bias_{app} != bias_{wth} - (separate biases to approach and withhold responding)
        
* Model Number: 5
* Parameters: \epsilon, \rho_{rew}, \rho_{pun}, bias_{app}, bias_{wth}
* Expected BIC: 4606
"""
def model_5(data, learning_rate, rho_rew, rho_pun, bias_app, bias_wth):
    """

    """
    def q_bias(q, state):   
        bias_matrix = np.array([
            [0.0, bias_app],  # Go+ (state 0): NoGo = 0, Go = bias_app
            [0.0, bias_wth],  # Go- (state 1): NoGo = 0, Go = bias_wth
            [0.0, 0.0],       # NoGo+ (state 2): No bias for either action
            [0.0, 0.0]        # NoGo- (state 3): No bias for either action
        ])
        return q[state] + bias_matrix[state] 
        
    def update_rule(q, state, action, reward):
        """
        q: np.ndarray
            The Q-values
        state: int
            The current state
        action: int
            The current action
        reward: int
            The reward
        """
        scaling_factor = (reward == 1) * rho_rew + (reward == -1) * rho_pun
        prediction_error = (scaling_factor * reward) - q[state, action]
        return q[state, action] + (learning_rate * prediction_error)
 
    
    return model_negative_log_likelihood(data, update_rule, q_bias=q_bias)

"""
Model-6: Assumes that:
    * \epsilon -  learning rate
    * \rho_rew^{app}, \rho_rew^{wth}, \rho_pun^{app}, \rho_pun^{wth} - feedback sensitivity
    * bias_{app}, bias_{wth} - biases:
"""
def model_6(data, learning_rate, 
                    rho_rew_app, rho_rew_wth, 
                    rho_pun_app, rho_pun_wth, 
                    bias_app, bias_wth):
    """
    """
    def q_bias(q, state):   
        bias_matrix = np.array([
            [0.0, bias_app],  # Go+ (state 0): NoGo = 0, Go = bias_app
            [0.0, bias_wth],  # Go- (state 1): NoGo = 0, Go = bias_wth
            [0.0, 0.0],       # NoGo+ (state 2): No bias for either action
            [0.0, 0.0]        # NoGo- (state 3): No bias for either action
        ])
        return q[state] + bias_matrix[state] 
        
    def update_rule(q, state, action, reward):
        """
        q: np.ndarray
            The Q-values
        state: int
            The current state
        action: int
            The current action
        reward: int
            The reward
        """
        # Define state-dependent feedback sensitivities
        rho = 1.0
        if state == 0:  # Go+ (approach block)
            rho = (reward == 1) * rho_rew_app + (reward == -1) * rho_pun_app
        elif state == 1:  # Go- (withdrawal block)
            rho = (reward == 1) * rho_rew_wth + (reward == -1) * rho_pun_wth
        else:
            # For NoGo+ (state 2) and NoGo- (state 3), no additional sensitivity
            rho = 1.0  # Default scaling factor, if no explicit sensitivity is applied

        # Compute prediction error
        prediction_error = (rho * reward) - q[state, action]
        return q[state, action] + (learning_rate * prediction_error)
    
    return model_negative_log_likelihood(data, update_rule, q_bias=q_bias)

"""
Model-7: Assumes that:
    * \epsilon_{rew}, \epsilon_{pun}, \epsilon_{omit} - learning rates
    * \rho_rew, \rho_pun - feedback sensitivity
    * bias_{app}, bias_{wth} - biases:
"""
def model_7(data, learning_rate_app, learning_rate_wth, rho_rew, rho_pun, bias_app, bias_wth):
    """
    """
    def q_bias(q, state):   
        bias_matrix = np.array([
            [0.0, bias_app],  # Go+ (state 0): NoGo = 0, Go = bias_app
            [0.0, bias_wth],  # Go- (state 1): NoGo = 0, Go = bias_wth
            [0.0, 0.0],       # NoGo+ (state 2): No bias for either action
            [0.0, 0.0]        # NoGo- (state 3): No bias for either action
        ])
        return q[state] + bias_matrix[state] 
        
    def update_rule(q, state, action, reward):
        """
        q: np.ndarray
            The Q-values
        state: int
            The current state
        action: int
            The current action
        reward: int
            The reward
        """
        # Determine the appropriate learning rate based on state
        if state in [0, 2]:  # Go+ or NoGo+ (approach conditions)
            learning_rate = learning_rate_app
        elif state in [1, 3]:  # Go- or NoGo- (withdrawal conditions)
            learning_rate = learning_rate_wth
        else:
            raise ValueError(f"Unexpected state: {state}")

        rho = (reward == 1) * rho_rew + (reward == -1) * rho_pun
        # Compute prediction error
        prediction_error = (rho * reward) - q[state, action]
        return q[state, action] + (learning_rate * prediction_error)
    
    return model_negative_log_likelihood(data, update_rule, q_bias=q_bias)

"""
Model-8: Assumes that:
    * \epsilon_{rew}, \epsilon_{pun}, \epsilon_{omit} - learning rates
    * \rho_rew, \rho_pun - feedback sensitivity
    * bias_{app}, bias_{wth} - biases:
    * bias_pav - Pavlovian bias
"""
def model_8(data, learning_rate_app, learning_rate_wth, rho_rew, rho_pun, bias_app, bias_wth, bias_pav):
    """
    """
    def q_bias(q, state):   
        bias_matrix = np.array([
            [0.0, bias_app],  # Go+ (state 0): NoGo = 0, Go = bias_app
            [0.0, bias_wth],  # Go- (state 1): NoGo = 0, Go = bias_wth
            [0.0, 0.0],       # NoGo+ (state 2): No bias for either action
            [0.0, 0.0]        # NoGo- (state 3): No bias for either action
        ])
       
        # Compute the Pavlovian bias term, p'_t(a_t)
        max_q = np.max(q[state])  
        pav_bias = np.zeros(2)   
        if max_q > 0:
            pav_bias[1] = bias_pav  
        elif max_q < 0:
            pav_bias[0] = bias_pav  
        
        return q[state] + bias_matrix[state] + pav_bias
        
    def update_rule(q, state, action, reward):
        """
        q: np.ndarray
            The Q-values
        state: int
            The current state
        action: int
            The current action
        reward: int
            The reward
        """
        # Determine the appropriate learning rate based on state
        if state in [0, 2]:  # Go+ or NoGo+ (approach conditions)
            learning_rate = learning_rate_app
        elif state in [1, 3]:  # Go- or NoGo- (withdrawal conditions)
            learning_rate = learning_rate_wth
        else:
            raise ValueError(f"Unexpected state: {state}")

        rho = (reward == 1) * rho_rew + (reward == -1) * rho_pun
        # Compute prediction error
        prediction_error = (rho * reward) - q[state, action]
        return q[state, action] + (learning_rate * prediction_error)
    
    return model_negative_log_likelihood(data, update_rule, q_bias=q_bias)


#%%
method = 'Nelder-Mead'  
#%%
# This optimization should work for the given data, 
# but feel free to try others as well, they might be faster.

# define a function to compute the BIC
@njit
def BIC(n, k, log_likelihood):
    """
    n: int
        The number of trials
    k: int
        The number of parameters
    log_likelihood: float   
        The log-likelihood of the model
    """
    retval= k * np.log(n) - 2 * log_likelihood
    print(f'BIC: {retval}')
    return retval

#%%
PARAMS = {
    'model_1': ['learning_rate', 'beta'],
    'model_2': ['learning_rate', 'rho_rew', 'rho_pun'],
    'model_3': ['learning_rate_rew', 'learning_rate_pun', 'learning_rate_omit', 'beta'],
    'model_4': ['learning_rate', 'beta', 'bias_app', 'bias_wth'],
    'model_5': ['learning_rate', 'rho_rew', 'rho_pun', 'bias_app', 'bias_wth'],
    'model_6': ['learning_rate', 'rho_rew_app', 'rho_rew_wth', 'rho_pun_app', 'rho_pun_wth', 'bias_app', 'bias_wth'],
    'model_7': ['learning_rate_app', 'learning_rate_wth',  'rho_rew', 'rho_pun', 'bias_app', 'bias_wth'],
    'model_8': ['learning_rate_app', 'learning_rate_wth',  'rho_rew', 'rho_pun', 'bias_app', 'bias_wth', 'bias_pav']
}

BOUNDS = {
    'model_1': {
        'learning_rate': (0, 1), 
        'beta': (0, 100) 
    },
    'model_2': {
        'learning_rate': (0, 1),
        'rho_rew': (0, 100),
        'rho_pun': (0, 100)
    },
    'model_3': {
        'learning_rate_rew': (0, 1),
        'learning_rate_pun': (0, 1),
        'learning_rate_omit': (0, 1),
        'beta': (0, 100)
    },
    'model_4': {
        'learning_rate': (0, 1),
        'beta': (0, 100),
        'bias_app': (-10, 10),
        'bias_wth': (-10, 10)
    }, 
    'model_5': {
        'learning_rate': (0, 1),
        'rho_rew': (0, 100),
        'rho_pun': (0, 100),
        'bias_app': (-10, 10),
        'bias_wth': (-10, 10)
    },
    'model_6': {
        'learning_rate': (0, 1),
        'rho_rew_app': (0, 100),
        'rho_rew_wth': (0, 100),
        'rho_pun_app': (0, 100),
        'rho_pun_wth': (0, 100),
        'bias_app': (-10, 10),
        'bias_wth': (-10, 10)
    },
    'model_7': {
        'learning_rate_app': (0, 1),
        'learning_rate_wth': (0, 1),
        'rho_rew': (0, 100),
        'rho_pun': (0, 100),
        'bias_app': (-10, 10),
        'bias_wth': (-10, 10)
    },
    'model_8': {
        'learning_rate_app': (0, 1),
        'learning_rate_wth': (0, 1),
        'rho_rew': (0, 100),
        'rho_pun': (0, 100),
        'bias_app': (-10, 10),
        'bias_wth': (-10, 10),
        'bias_pav': (-10, 10)
    }
}

INITIAL_PARAMS = {
    'model_1': {
        'learning_rate': 0.1,
        'beta': 0.5
    },
    'model_2': {
        'learning_rate': 0.1,
        'rho_rew': 0.5,
        'rho_pun': 0.5
    },
    'model_3': {
        'learning_rate_rew': 0.1,
        'learning_rate_pun': 0.1,
        'learning_rate_omit': 0.1,
        'beta': 0.5
    }, 
    'model_4': {
        'learning_rate': 0.1,
        'beta': 0.5,
        'bias_app': 0.5,
        'bias_wth': 0.5
    }, 
    'model_5': {
        'learning_rate': 0.1,
        'rho_rew': 0.5,
        'rho_pun': 0.5,
        'bias_app': 0.5,
        'bias_wth': 0.5
    },
    'model_6': {
        'learning_rate': 0.1,
        'rho_rew_app': 0.5,
        'rho_rew_wth': 0.5,
        'rho_pun_app': 0.5,
        'rho_pun_wth': 0.5,
        'bias_app': 0.5,
        'bias_wth': 0.5
    },
    'model_7': {
        'learning_rate_app': 0.1,
        'learning_rate_wth': 0.1,
        'rho_rew': 0.5,
        'rho_pun': 0.5,
        'bias_app': 0.5,
        'bias_wth': 0.5
    },
    'model_8': {
        'learning_rate_app': 0.1,
        'learning_rate_wth': 0.1,
        'rho_rew': 0.5,
        'rho_pun': 0.5,
        'bias_app': 0.5,
        'bias_wth': 0.5,
        'bias_pav': 0.5
    }
}

MODELS = {
    'model_1': model_1,
    'model_2': model_2,
    'model_3': model_3,
    'model_4': model_4,
    'model_5': model_5,
    'model_6': model_6,
    'model_7': model_7,
    'model_8': model_8
}

#%%
"""
Optimize the models: 
    - fitting all the parameters of each model to each individual subject, 
    - using the scipy minimize function. 

Pay attention to initialize the parameters 
    - to reasonable values 
    - set sensible bounds for each parameter 
        (since Q-values get turned into probabilities 
            through a softmax, which uses an exponential function, 
            you may have to limit some of the parameters to certain magnitudes, 
            to prevent overflow errors). 
    - Given the number of models this can take some minutes, 
        to save time you can e.g.  only apply the logarithm at the end, 
        rather than during every iteration of your for-loop
"""
def fit_subject(subject_id, model_id, df, model, method='Nelder-Mead', 
                initial_params=INITIAL_PARAMS, bounds=BOUNDS):
        
    subject_data = subject_df(df, subject_id) # subset data to one subject
    subject_data = subject_data.reset_index(drop=True)  # not resetting the index can lead to issues
    
    print(f'Fitting model {model_id} to subject {subject_id}')

    # Define yourself a loss for the current model
    def loss(params):
        return model(subject_data, *params)

    opt_initial_params = [initial_params[model_id][p] for p in PARAMS[model_id]]
    opt_bounds = [bounds[model_id][p] for p in PARAMS[model_id]]
    
    res = minimize(loss, opt_initial_params, 
                            bounds=opt_bounds, 
                            method=method, 
                            tol=1e-6, 
                            options={'disp': True })

    num_params = len(res.x)
    num_trials = len(subject_data)
    
    return { 
              "subject_id": subject_id, 
              "model_id": model_id, 
              "negative_log_likelihood": res.fun, 
              "params": res.x, 
              "num_params": num_params, 
              "num_trials": num_trials,
        }

#%%
# Pick one model to start with
def fit_model(df, model, model_id, method='Nelder-Mead', use_cache=True, 
              initial_params=INITIAL_PARAMS, bounds=BOUNDS):
    
    if os.path.exists(f'{model_id}_model_results.pkl') and use_cache:
        print(f'Loading cached results for model {model_id}')
        with open(f'{model_id}_model_results.pkl', 'rb') as f:
            model_results = pickle.load(f)
        return model_results
    
    # Loop over all subjects
    subject_ids = np.sort(np.unique(df.ID))
    # Parallel processing with Joblib
    # subject_data = subject_data.reset_index(drop=True)  # not resetting the index can lead to issues
    subject_results = Parallel(n_jobs=-1)(delayed(fit_subject)(subject_id, model_id , df, model, method=method, initial_params=initial_params, bounds=bounds) for subject_id in subject_ids)

    # Collect the results 
    for subject_result in subject_results:
        subject_id = subject_result['subject_id']
        model_id = subject_result['model_id']
        negative_log_likelihood = subject_result['negative_log_likelihood']
        params = subject_result['params']
        print(f"model {model_id} subject {subject_id}: params = {params}, negative-log-likelihood = {negative_log_likelihood}")
                
    # compute BIC
    subject_bics = [BIC(r["num_trials"], r["num_params"], -r["negative_log_likelihood"]) for r in subject_results]
    subject_neg_log_likelihoods = [r["negative_log_likelihood"] for r in subject_results]
    model_bic = np.sum(subject_bics)
    model_neg_log_likelihood = np.sum(subject_neg_log_likelihoods)
   
    model_result = { 
            "model_id": model_id, 
            "model_neg_log_likelihood": model_neg_log_likelihood, 
            "model_bic": model_bic , 
            "subject_results": subject_results 
    }
   
    with open(f'{model_id}_model_results.pkl', 'wb') as f:
        pickle.dump(model_result, f)
        
    return model_result

def fit_models(df, models, method='Nelder-Mead', use_cache=True):
    model_results_map = Parallel(n_jobs=-1)(delayed(fit_model)(df, model, model_id, method, use_cache=use_cache) for model_id, model in models.items())
        
    for model_result in model_results_map:
        model_id = model_result["model_id"]
        neg_log_likelihood = model_result["model_neg_log_likelihood"] 
        bic = model_result["model_bic"]
        
        print(f'Model [{model_id}] neg_Log-Likelihood: {neg_log_likelihood}')
        print(f'Model [{model_id}] BIC: {bic}')
        
    return {  mr["model_id"]: mr for mr in model_results_map }

def save_mode_results(model_results):
    """
    Save the model results to a JSON file, pretty
    """
    #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #save_path = f'model_results_{timestamp}.pkl'
    save_path = 'model_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(model_results, f)

#%%
model_results = fit_models(df, MODELS, method='Nelder-Mead', use_cache=True)
save_mode_results(model_results)
#%%
model_results_df = pd.DataFrame()

for model_id, model_result in model_results.items():
    model_results_df = pd.concat([model_results_df, pd.DataFrame(model_result)])

model_results_df    
#%%
def min_neg_log_likelihood(model_results):
    """
    Find the model with the minimum negative log-likelihood.
    """
    min_model = None
    min_neg_log_likelihood = np.inf
    for model_id, model_result in model_results.items():
        if model_result['model_neg_log_likelihood'] < min_neg_log_likelihood:
            min_model = model_id
            min_neg_log_likelihood = model_result['model_neg_log_likelihood']
    return min_model

def min_bic(model_results):
    min_model = None
    min_bic = np.inf
    for model_id, model_result in model_results.items():
        if model_result['model_bic'] < min_bic:
            min_model = model_id
            min_bic = model_result['model_bic']
    return min_model
   
min_neg_log_likelihood = min_neg_log_likelihood(model_results)
min_bic = min_bic(model_results) 
print(f'Minimum Negative Log-Likelihood: {min_neg_log_likelihood}')
print(f'Minimum BIC: {min_bic}')
#%%
assert len(model_results.items()) == len(MODELS)
#assert np.isclose(model_results['model_1']['model_neg_log_likelihood'] , 3248.52)
#assert np.isclose(model_results['model_1']['model_bic'] , -6369.10.98)

#%%
"""
\subsection{Task-5:}

- Sum up the optimized log-likelihoods across all subjects for each model ?

- Use this and all other relevant values to compute the BIC score for each model ?
  (using e.g. the BIC equation of Wikipedia). 
  
- What does this tell you about which model describes the data best ?
"""
def load_model_results(file_path='model_results.pkl'):
    """
    Load model results from pkl file.
    """
    with open(file_path, 'rb') as f:
        model_results = pickle.load(f)
    return model_results

model_results = load_model_results()

def plot_log_likelihoods(models, log_likelihoods, 
                         save_path='log_likelihoods.png'):
    """
    Plot the log-likelihoods for each model.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(models, log_likelihoods)
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Negative Log-Likelihood for each Model')
    plt.savefig(save_path)
    plt.show()

def plot_bics(models, bics, save_path='bics.png'):
    """
    Plot the BICs for each model.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(models, bics)
    plt.ylabel('BIC')
    plt.title('BIC for each Model')
    plt.savefig(save_path)
    plt.show()

# Plot the log-likelihoods and BICs
models = list(model_results.keys())
log_likelihoods = [ -model_results[model_id]['model_neg_log_likelihood'] for model_id in models ]
bics = [ model_results[model_id]['model_bic'] for model_id in models ]

plot_log_likelihoods(models, log_likelihoods)
plot_bics(models, bics)

#%%
# Plot learning rates of the last model.
#%%
def plot_nll_and_bic(models, nlls, bics, save_path='nll_and_bic.png'):
    """
    Plot Negative Log-Likelihood (NLL) and BIC for each model in one graph.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    padding =10

    log_likelihoods = [-nll for nll in nlls]    
    # Adding some padding
    nll_min = min(log_likelihoods) - padding   
    nll_max = max(log_likelihoods) + padding
  
    ax1.bar(models, log_likelihoods, color='blue', alpha=0.7, label='Log-Likelihood')
    ax1.set_ylabel('Log-Likelihood', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(nll_min, nll_max)

    # Plot BIC on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(models, bics, color='red', marker='o', linestyle='-', label='BIC')
    ax2.set_ylabel('BIC', color='red')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Add titles and legends
    plt.title('Log-Likelihood and BIC for each Model')
    fig.tight_layout()
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)
    plt.savefig(save_path)
    plt.show()

# Data Preparation
models = list(model_results.keys())
nlls = [model_results[model_id]['model_neg_log_likelihood'] for model_id in models]
bics = [model_results[model_id]['model_bic'] for model_id in models]

# Plot NLL and BIC in a single graph
plot_nll_and_bic(models, nlls, bics)
#%%
"""
\subsection{Task-6}:  
    \textbf{Compare the fitted $\epsilon_{\text{app}}$ and $\epsilon_{\text{wth}}$ for the last model.} \\
    How do you interpret the difference in their means?
"""
subject_results = model_results['model_8']['subject_results']
median_eps_app = np.median([sr['params'][0] for sr in subject_results])
median_eps_wth = np.median([sr['params'][1] for sr in subject_results])
median_bias_app = np.median([sr['params'][2] for sr in subject_results])
median_bias_wth = np.median([sr['params'][3] for sr in subject_results])

for sr in subject_results:
    print(f'Subject ID: {sr["subject_id"]}')
    for idx, param in enumerate(PARAMS['model_8']):
        print(f'\t\t{param}: {sr["params"][idx]:.5f}')
print("")
print(f'Median Learning Rate for Approach: {median_eps_app:.5f}')
print(f'Median Learning Rate for Withdrawal: {median_eps_wth:.5f}')
print(f'Median Bias for Approach: {median_bias_app:.5f}')
print(f'Median Bias for Withdrawal: {median_bias_wth:.5f}')
print("")

# Median Learning Rate for Approach: 0.19933
# Median Learning Rate for Withdrawal: 0.25130
# Median Bias for Approach: 1.78337
# Median Bias for Withdrawal: 1.60453

#%%
"""
%% Task-7:

\textbf{Bonus: Fit the first subject 10 times with the last model, 
               using different initial parameters.} \\
                   
Create a scatter plot between the fitted 
    $\text{bias}_{\text{app}}$ and $\text{bias}_{\text{wth}}$ 
    across the fits. 
    
How do you explain this plot?

For the last model:
    - Compare the fitted :
        - \epsilon_{app} 
        - \epsilon_{wth} 
        
    - How do you interpret the difference in their means ?
"""
#%%
def random_initialization_fit_subject(subject_id=0,model_id='model_8', model=model_8, df=df,  n_iter=10):
    param_list=[]
    for i in range(n_iter):
        print(f'Fitting model_8 to subject {subject_id}, iteration {i}')
        bounds_by_param = [BOUNDS[model_id][p] for p in PARAMS[model_id]]
        random_initial_params = {model_id: {
            p: np.random.uniform(*bounds_by_param[idx]) 
                for idx, p in enumerate(PARAMS[model_id])
        }}
        param_list.append(random_initial_params)
        
    # model_results_map = fit_subject(subject_id, model_id, df, model, initial_params=param_list[0])
    model_results_map = Parallel(n_jobs=-1)(delayed(fit_subject)(subject_id, model_id, 
                                                                 df, model, 
                                                                 initial_params=initial_params) 
                                            for _, initial_params in enumerate(param_list))
    return model_results_map

#%%
def scatter_plot_bias_approach_bias_with(use_cache=True, model_id='model_8'):
    if os.path.exists('random_initialization_model_result_map.pkl') and use_cache:
        random_initialization_model_result_map = pickle.load(open('random_initialization_model_result_map.pkl', 'rb'))
    else:
        random_initialization_model_result_map = random_initialization_fit_subject(df=df, n_iter=30)
    with open('random_initialization_model_result_map.pkl', 'wb') as f:
        pickle.dump(random_initialization_model_result_map, f)
    found_params = [random_initialization_model_result_map[i]['params'] for i,_ in  enumerate(random_initialization_model_result_map)] 
    bias_app_idx = PARAMS[model_id].index('bias_app')
    bias_wth_idx = PARAMS[model_id].index('bias_wth')
    bias_app = [p[bias_app_idx] for p in found_params]
    bias_wth = [p[bias_wth_idx] for p in found_params]
    plt.figure(figsize=(10, 10))
    plt.title('Scatter Plot of Bias for Approach and Withhold')
    plt.xlabel('Bias for Approach')
    plt.ylabel('Bias for Withhold')
    bias_app_with_df = pd.DataFrame({'bias_app': bias_app, 'bias_wth': bias_wth})
    sns.scatterplot(data=bias_app_with_df, x='bias_app', y='bias_wth')
    plt.grid(True)  
    plt.savefig('bias_app_with_scatter.png')
    plt.show()
#%%
scatter_plot_bias_approach_bias_with()
#%%
