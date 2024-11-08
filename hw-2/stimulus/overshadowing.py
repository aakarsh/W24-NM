import numpy as np

def setup_stimuli(num_trials, num_stimuli, 
                    pre_train_period=(0, .50), 
                    train_period=(.50, 1.0), 
                    seed=0):
    """
        Pre-Training: - 
        Training: s_1 + s_2 -> r, 
        Result: s_1 -> \alpha_1' r and s_2 -> \alpha_2 r 
    """
    np.random.seed(seed)
    stimuli = np.zeros((num_stimuli, num_trials)) 
    rewards = np.zeros(num_trials)
    
    # pre-training - no pretraining
    pre_train_start = int(num_trials * pre_train_period[0])
    pre_train_end = int(num_trials * pre_train_period[1]) 
    
    # training
    train_start = int(num_trials * train_period[0])
    train_end = int(num_trials * train_period[1])
    
    # TODO Is this right ?  
    for i in range(train_start, train_end):
        for j in range(num_stimuli):
            stimuli[j, i] = 1
        rewards[i] = 1
    
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
