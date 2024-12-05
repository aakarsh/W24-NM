"""
    Example structure for fitting multiple models, feel free to modify to your liking
"""
#%%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sns.set_theme(style='white', context='notebook', font_scale=1.2)
#%%
df = pd.read_csv('gen_data.csv')
cue_mapping = {1: 'Go+', 2: 'Go-', 3: 'NoGo+', 4: 'NoGo-'}  # Go+ = Go to win, Go- = go to avoid losing, NoGo+ = don't go to win, NoGo- = don't go to avoid losing
#%%
"""

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
    (1) - reward was delivered  
    (0) - nothing was delivered 
    (-1)-  a punishment was given 
"""
#%%
# Exercise-1: Plot the Accuracy for each Cue
"""
    Recreate figure 2E of the paper :
    "
    Go and no-go learning in reward and punishment: 
        Interactions between affect and effect” 
        
    Only the bar plots are important here, no need for error 
    bars or significance tests.
"""
# TODO
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
            # Subject answers presses the button to avoid punishment 
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

def num_outcomes(subject_df):
    return len([1, 0, -1])

def empty_q(num_cues, num_actions):
    return np.zeros((num_cues, num_actions))

#%%
first_subject_df = subject_df(df, 0)
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
    # Run a q-learning model on the data,  return the 
    # log-likelihood parameters are learning rate and beta, 
    # the feedback sensitivity.
    q = empty_q(num_states(data), num_actions(data))
    
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
        prob = softmax(q[state])
        
        # Compute the log-likelihood
        log_likelihood += np.log(prob[action])
         
        # Update the Q-values - feedback sensitive prediction error.
        prediction_error = (beta * reward) - q[state, action] 
        q[state, action] += learning_rate * prediction_error
         
    return -log_likelihood

"""
Model-2: Assumes that:
    * \epsilon - learning rate
    * \rho_pun ,\rho_rew - feedback sensitivity
        * \rho_pun != \rho_rew - (separate reward and punishment sensitivities) 
    * No bias parameters
        * bias_{app} = bias_{wth} = 0 
"""

"""
Model-3: Assumes that:
    * \epsilon_new, \epsilon_rew - learning rates
    * \beta - feedback sensitivity
    * No bias parameters
        * bias_{app} = bias_{wth} = 0
"""

"""
Model-4: Assumes that:
    * \epsilon - learning rate
    * \beta - feedback sensitivity
    * Biases: 
        * bias_{app} != bias_{wth} - (separate biases to approach and withhold responding)
"""

"""
Model-5: Assumes that:
    * \epsilon - learning rate
    * \rho_pun ,\rho_rew - feedback sensitivity
    * bias_{app}, bias_{wth} - biases: 
        * bias_{app} != bias_{wth} - (separate biases to approach and withhold responding)
"""

"""
Model-6: Assumes that:
    * \epsilon -  learning rate
    * \rho_rew^{app}, \rho_rew^{wth}, \rho_pun^{app}, \rho_pun^{wth} - feedback sensitivity
    * bias_{app}, bias_{wth} - biases:
"""

"""
Model-7: Assumes that:
    * \epsilon_{app}, \epsilon_{wth} - learning rates
    * \rho_rew, \rho_pun - feedback sensitivity
    * bias_{app}, bias_{wth} - biases:
"""

#%%
method = 'Nelder-Mead'  
# This optimization should work for the given data, 
# but feel free to try others as well, they might be faster.

# define a function to compute the BIC
def BIC(...):
    return ...

#%%
for j, learner in enumerate([model_1]):

    for i, subject in enumerate(np.unique(df.ID)):
        subject_data = ... # subset data to one subject
        subject_data = subject_data.reset_index(drop=True)  # not resetting the index can lead to issues

        if j == 0:

            # define yourself a loss for the current model
            def loss(params):
                return ...
            res = minimize(loss, ...initial_params..., bounds=..., method=method)

            # save the optimized log-likelihhod

            # save the fitted parameters

    # compute BIC


# Plot learning rates of the last model.


# Bonus
