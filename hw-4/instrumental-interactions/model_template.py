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
def plot_cue_accuracy(df, cue_mapping=cue_mapping, save_path='cue_accuracy.png'):
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
# Note that Don't go to win has lowest accuracy, 
# while Go to Win has highest accuracy
plot_cue_accuracy(df)
#%%

"""
Program the log likelihood functions of the models 1 to 7 
(including) presented in 
    ”Disentangling the Roles of Approach, Activation and 
    Valence in Instrumental and Pavlovian Responding” 
    
    (see Table 2 of that paper for the model numbering and 
    relevant parameters). 
    
    The paper uses these parameters 
            - Learning Rate ε
            - Feedback Sensitivity β
                - The general feedback sensitivity β 
                can be replaced by separate reward and punishment sensitivities ρ 
                (we don't include a sensitivity for omission) 
                - There can be different learning rates ε 
                for:
                    - Reward, 
                    - Feedback Omission/no-reward-no-punish, 
                    - Punishment 
                        (The paper doesn't make use of omissions, 
                        so they use only two learning rates, 
                        you will need three.)
                    - There can be a: 
                        - General bias to approach: $bias_{app}$
                        - General bias to withhold responding: $bias_{wth}$
"""
# Define yourself a softmax function
def softmax(x):
    # TODO:
    return np.exp(x) / np.sum(np.exp(x), axis=0) 

def model_1(data, learning_rate, beta):
    # Run a q-learning model on the data,  return the 
    # log-likelihood parameters are learning rate and beta, 
    # the feedback sensitivity.
    q = np.zeros(...)
    log_likelihood = ...

    for i in range(len(data)):
        ...

    return ...


#%%
method = 'Nelder-Mead'  # this optimization should work for the given data, but feel free to try others as well, they might be faster

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


# plot learning rates of the last model


# Bonus


    

