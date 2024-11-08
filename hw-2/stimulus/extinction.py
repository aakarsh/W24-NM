#%%
import numpy as np


"""

Design an appropriate array to describe the belief of the animal
about the current state of the environment on each trial, right
before receiving the CS. 
(Hint: How many states do you need?  When do they change?) 
The animal first encounters 50 conditioning trials, followed by 
50 extinction trials the next day, and, after a 30-day delay, 
encounters the CS again for a single trial.

"""
#%%
conditioning_trials = 50
extinction_trials = 50
delay = 30
num_states = 2


#%%
# animal creates a state representation for stimulus such that
# it appends a new set of states to the to keep track of newly 
# encountered stimuli state
# CS -> (stimulus) -> Shock
# B -> 

state_array = np.array([[], [],[]])

# CS - Conditioned stimulus - Tone
# US - Unconditioned stimulus - Shock
"""
 Write a function that infers the belief array using a 
 simple heuristic. 
 
 This function should take three inputs: 
    - the state of the previous trial, 
    - the similarity of the previous trial to the one 
        before it in terms of observations, 
    - the time since the last trial. 
    
    Based on these arguments, return a belief/probability over 
    all states under consideration. 
    
    Assume 100\% certainty of being in state 1 on the 
    first trial.
"""
def state_heuristic(previous_state_idx, 
                    prior_similarity, time_since_last_trial, 
                    state_array=state_array):
    """
    
    """
    num_states = len(state_array)
    if time_since_last_trial > 30:
        # uniform belief over all states
        return np.ones(num_states) / num_states  
            
    if prior_similarity > 0.5:
        state_beliefs = np.zeros(num_states)
        state_beliefs[previous_state_idx] = 1
        return state_beliefs 
    else: # unknown state.
        return np.ones(num_states) / num_states  


"""
Maintain a learned association strength between CS and US for
each state and update it after each trial according to the 
Rescorla-Wagner rule. 

Weight the update magnitude by the belief in the current state 
and assume the belief remains constant throughout the trial.
"""



"""
Maintain a learned association strength between CS and US for
each state and update it after each trial according to the 
Rescorla-Wagner rule. Weight the update magnitude by the belief 
in the current state and assume the belief remains constant 
throughout the trial.
"""

# %%
