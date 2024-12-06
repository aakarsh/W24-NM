#%%
"""
    Example structure for fitting multiple models, feel free to modify to your liking
"""
from numba import njit
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import multiprocessing as mp
from joblib import Parallel, delayed

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

def num_outcomes(subject_df):
    return len([1, 0, -1])

def empty_q(num_cues, num_actions):
    return np.zeros((num_cues, num_actions))

#%%
first_subject_df = subject_df(df, 0)

#%%
def model_negative_log_likelihood(data, q_update):
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
        # Update the Q-values - Feedback sensitive prediction error.
        q[state, action] = q_update(q, state, action, reward)
        
    return -log_likelihood

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
        if reward == 1:
            prediction_error = (rho_rew * reward) - q[state, action]
        elif reward == -1:
            prediction_error = (rho_pun * reward) - q[state, action]
        elif reward == 0: # Omission
            prediction_error = 0 - q[state, action]
        else:
            raise ValueError(f'Invalid reward value: {reward}')
        return q[state, action] + learning_rate * prediction_error
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
        if reward == 1:
            prediction_error = (beta * reward) - q[state, action]
            return q[state, action] + learning_rate_rew * prediction_error
        elif reward == -1:
            prediction_error = (beta * reward) - q[state, action]
            return q[state, action] + learning_rate_pun * prediction_error
        elif reward == 0: # Omission
            prediction_error = 0 - q[state, action]
            return q[state, action] + learning_rate_omm * prediction_error
        
    return model_negative_log_likelihood(data, update_rule)  

"""
Model-4: Assumes that:
    * \epsilon - learning rate
    * \beta - feedback sensitivity
    * Biases: 
        * bias_{app} != bias_{wth} - (separate biases to approach and withhold responding)
Model Number: 4
Parameters: \epsilon, \beta, bias_{app}, bias_{wth}
"""
def model_4(data, learning_rate, beta, bias_app, bias_wth):
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
        prediction_error = (beta * reward) - q[state, action]
        
        return False #q[state, action] + learning_rate * prediction_error + bias_app - bias_wth
    
    return model_negative_log_likelihood(data, update_rule)
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
        if reward == 1:
            prediction_error = (rho_rew * reward) - q[state, action]
        elif reward == -1:
            prediction_error = (rho_pun * reward) - q[state, action]
        return q[state, action] + learning_rate * prediction_error + bias_app - bias_wth
    
    return model_negative_log_likelihood(data, update_rule)
"""
Model-6: Assumes that:
    * \epsilon -  learning rate
    * \rho_rew^{app}, \rho_rew^{wth}, \rho_pun^{app}, \rho_pun^{wth} - feedback sensitivity
    * bias_{app}, bias_{wth} - biases:
"""

"""
Model-7: Assumes that:
    * \epsilon_{rew}, \epsilon_{pun}, \epsilon_{omit} - learning rates
    * \rho_rew, \rho_pun - feedback sensitivity
    * bias_{app}, bias_{wth} - biases:
"""

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
    return k * np.log(n) - 2 * log_likelihood

#%%
PARAMS = {
    'model_1': ['learning_rate', 'beta'],
    'model_2': ['learning_rate', 'rho_rew', 'rho_pun'],
    'model_3': ['learning_rate_rew', 'learning_rate_pun', 'beta'],
    'model_4': ['learning_rate', 'beta', 'bias_app', 'bias_wth'],
    'model_5': ['learning_rate', 'rho_rew', 'rho_pun', 'bias_app', 'bias_wth'],
    'model_6': ['learning_rate', 'rho_rew_app', 'rho_rew_wth', 'rho_pun_app', 'rho_pun_wth', 'bias_app', 'bias_wth'],
    'model_7': ['learning_rate_rew', 'learning_rate_pun', 'learning_rate_omit', 'rho_rew', 'rho_pun', 'bias_app', 'bias_wth']
}

BOUNDS = {
    'model_1': {
        'learning_rate': (0, 1),
        'beta': (0, 1)
    }
} 

INITIAL_PARAMS = {
    'model_1': {
        'learning_rate': 0.1,
        'beta': 0.5
    },
}

MODELS = {
    'model_1': model_1,
    # 'model_2': model_2,
    # 'model_3': model_3,
    # 'model_4': model_4,
    'model_5': model_5,
    #'model_6': model_6,
    #'model_7': model_7
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
def fit_subject(subject_id, model_id, df, model, method='Nelder-Mead'):
    subject_data = subject_df(df, subject_id) # subset data to one subject
    subject_data = subject_data.reset_index(drop=True)  # not resetting the index can lead to issues
    print(f'Fitting model {model_id} to subject {subject_id}')

    # define yourself a loss for the current model
    def loss(params):
        return model(subject_data, *params)
    
    initial_params = [INITIAL_PARAMS[model_id][p] for p in PARAMS[model_id]]
    bounds = [BOUNDS[model_id][p] for p in PARAMS[model_id]]
    res = minimize(loss, initial_params, bounds=bounds, method=method, 
                    tol=1e-6,
                    options={'disp': True })

    num_params = len(res.x)
    num_trials = len(subject_data)
    #np.save(f'{subject_id}, {model_id}, log_likelihoods_model_{j}_subject_{i}', res.x)
    # save the fitted parameters
    return subject_id, model_id, res.fun, res.x, num_params, num_trials

#%%
# Pick one model to start with
def fit_model(df, model, model_id, method='Nelder-Mead'):
    # Loop over all subjects
    subject_ids = np.sort(np.unique(df.ID))
    # Parallel processing with Joblib
    # subject_data = subject_data.reset_index(drop=True)  # not resetting the index can lead to issues
    results = Parallel(n_jobs=-1)(delayed(fit_subject)(subject_id, model_id , df, model) for subject_id in subject_ids)

    # Collect the results 
    subject_bics = []
    log_likelihoods = []
    for subject_index, model_id, negative_log_likelihood, params, num_params, trial_count in results:
        print(f"subject {subject_index+1}: params = {params}, negative-log-likelihood = {negative_log_likelihood}")
        subject_bics.append(BIC(trial_count, num_params, -negative_log_likelihood))
        log_likelihoods.append(-negative_log_likelihood)
        
    # compute BIC
    model_bic = np.sum(subject_bics)
    model_log_likelihood = np.sum(log_likelihoods)

    return model_id, model_log_likelihood, model_bic  
        
def fit_models(df, models, method='Nelder-Mead'):
    model_results = Parallel(n_jobs=-1)(delayed(fit_model)(df, model, model_id, method) for model_id, model in models.items())
    
    # Collect model results
    model_results_map = {model_id: { 'log_likelihood': log_likelihood, 'bic': bic} 
                            for model_id, log_likelihood, bic in model_results }
    
    for model_id, log_likelihood, bic in model_results:
        print(f'Model [{model_id}] Log-Likelihood: {log_likelihood}')
        print(f'Model [{model_id}] BIC: {bic}')
        
    return model_results_map

#%%
model_results = fit_models(df, MODELS, method='Nelder-Mead')

assert len(model_results) == len(MODELS)

assert np.isclose(model_results['model_1']['log_likelihood'] , -3248.52)
assert np.isclose(model_results['model_1']['bic'] , 6624.98)
#%%
"""
- Sum up the optimized log-likelihoods across all subjects for each model ?

- Use this and all other relevant values to compute the BIC score for each model ?
  (using e.g. the BIC equation of Wikipedia). 
  
- What does this tell you about which model describes the data best ?
"""
def plot_log_likelihoods(models, log_likelihoods, save_path='log_likelihoods.png'):
    plt.figure(figsize=(10, 5))
    plt.bar(models, log_likelihoods)
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Negative Log-Likelihood for each Model')
    plt.savefig(save_path)
    plt.show()

def plot_bics(models, bics, save_path='bics.png'):
    plt.figure(figsize=(10, 5))
    plt.bar(models, bics)
    plt.ylabel('BIC')
    plt.title('BIC for each Model')
    plt.savefig(save_path)
    plt.show()

# Plot the log-likelihoods and BICs
models = list(model_results.keys())
log_likelihoods = [-model_results[model_id]['log_likelihood'] for model_id in models]
bics = [model_results[model_id]['bic'] for model_id in models]
plot_log_likelihoods(models, log_likelihoods)
plot_bics(models, bics)

#%%
"""
for the last model:
    - compare the fitted :
        - \epsilon_{app} 
        - \epsilon_{wth} 
    - How do you interpret the difference in their means?
"""

# Plot learning rates of the last model.


# Bonus
