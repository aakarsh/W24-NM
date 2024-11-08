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
conditioning_trials = 50
extinction_trials = 50
delay = 30
num_states = 3



# animal creates a state representation for stimulus such that
# it appends a new set of states to the to keep track of newly 
# encountered stimuli state
# CS -> (stimulus) -> Shock
# B -> 
state_array = np.array([[], [],[]])
state_index_map = { 0: 0, 1: 1, 2: 2 }

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
def state_heuristic(previous_state, 
                    prior_similarity, time_since_last_trial, 
                    state_array=state_array):
    """
    
    """
    num_states = len(state_array)
    state_beliefs = np.zeros(num_states)
    if time_since_last_trial > 30:
        state_beliefs[-1] = 1
        return state_beliefs # Assume last state is uncertain state.
    
    if prior_similarity > 0.5:
        index_of_state = state_index_map[previous_state]
        state_beliefs[index_of_state] = 1
        return state_beliefs 
    else:
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
