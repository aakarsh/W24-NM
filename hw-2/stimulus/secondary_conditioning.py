import numpy as np

def setup_stimuli(num_trials, num_stimuli, 
                    pre_train_period=(0, .50), 
                    train_period=(.50, 1.0), 
                    seed=0):
    """
    Pre-Training: s_1 -> r
    Training: s_2 -> s_1 
    Result: s_2 -> r 
    """
    np.random.seed(seed)
    stimuli = np.zeros((num_stimuli, num_trials))
    rewards = np.zeros(num_trials)
    
    # pre-training 
    pre_train_start = int(num_trials * pre_train_period[0])
    pre_train_end = int(num_trials * pre_train_period[1])
    for i in range(pre_train_start, pre_train_end):
        stimuli[0, i] = 1
        rewards[i] = 1
    
    # training
    train_start = int(num_trials * train_period[0])
    train_end = int(num_trials * train_period[1])

    for i in range(train_start, train_end):
        stimuli[1, i] = 1
        stimuli[0, i] = 1
        rewards[i] = 0
       
    idealized_expected_rewards = np.zeros(num_trials) 
    return {
        "stimuli": stimuli,
        "rewards": rewards,
        "train_start": train_start,
        "train_end": train_end,
        "pre_train_start": pre_train_start,
        "pre_train_end": pre_train_end,
        "idealized_expected_rewards": idealized_expected_rewards,
    }