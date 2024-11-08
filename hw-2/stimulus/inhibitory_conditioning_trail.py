import numpy as np



def setup_stimuli(num_trials, num_stimuli, 
                        pre_train_period=(0, .20), 
                        train_period=(.20, .40), ideal_learning_rate=0.3):
    """
    Inhibitory Effect:
    
    Pre-Training: - 
    Training: s_1 + s_2 -> ., s_1 -> r
    Ideal Result: s_1 -> r, s_2 -> -r 
    """
    stimuli = np.zeros((num_stimuli, num_trials))
    rewards = np.zeros(num_trials)
    idealized_expected_rewards = np.zeros(num_trials)
   
    # pre-training
    pre_train_start = int(num_trials * pre_train_period[0])
    pre_train_end = int(num_trials * pre_train_period[1])
    pretrain_stimulus_idx = 0

    # - Nothing happens 
    prev_expected_reward = 0
    for trial_idx in range(pre_train_start, pre_train_end):
        # During pre-training always show the primary stimulus
        stimuli[pretrain_stimulus_idx, trial_idx] = 0
        rewards[trial_idx] = 0
        
        # During pre-training we assume full saturation is reached for the 
        # during the pre-training period.
        # incremental_association = 1.0 / ( pre_train_end - pre_train_start )
        prediction_error = rewards[trial_idx] - prev_expected_reward 
        idealized_expected_rewards[trial_idx]  = prev_expected_reward + (ideal_learning_rate * prediction_error)
        idealized_expected_rewards[trial_idx] = min(idealized_expected_rewards[trial_idx], 1.0)
        prev_expected_reward = idealized_expected_rewards[trial_idx]
        
    # In training, multiple stimuli are presented and reward is given
    # off by one ? 
    assert pre_train_end == int(num_trials * train_period[0])
    train_start = pre_train_end-1 
    train_end   = int(num_trials * train_period[1])
    assert pre_train_end == train_start+1
    inhibition_trials = []
    # s_1 + s_2 -> . , s_1 ->  r 
    for trial_idx in range(train_start, train_end):
        # During training show both the primary and other stimuli.
        # Present all stimuli
        is_inhibition_trial = trial_idx % 2 == 0
        inhibition_trials.append(is_inhibition_trial)
       
        if is_inhibition_trial:
            # s_1 + s_2 -> . 
            for stimulus_idx in range(num_stimuli):
                stimuli[stimulus_idx, trial_idx] = 1
            rewards[trial_idx] = 0
        else:
            # s_1 -> r
            stimuli[0, trial_idx] = 1
            rewards[trial_idx] = 1
           
    # TODO: Add idealized test rewards.
    idealized_expected_rewards = np.zeros(num_trials) 
    # create stimuli, rewards presented in pre-training, 
    # training periods expected rewards in test period.
    return {
            "stimuli": stimuli, 
            "rewards": rewards, 
            "train_start": train_start,
            "train_end": train_end,
            "pre_train_start": pre_train_start,
            "pre_train_end": pre_train_end,
            "idealized_expected_rewards": idealized_expected_rewards,
            # TODO: idealized expected rewards.
        }
 