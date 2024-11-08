import numpy as np
import random

"""
Blocking is the paradigm that first led to the suggestion of the 
Rescorla-Wagner rule. 
In blocking, during the pre-training period, 
a stimulus is associated with a reward, as in Pavlovian conditioning. 
Then during the training period, a second stimulus is presented along with 
the first, in association with the same reward. In this case, the 
pre-existing association of the first stimulus with the reward blocks an 
association from forming between the second stimulus and the reward. 

Thus, after training, a conditioned response is evoked only by the 
first stimulus, not by the second. 

This follows from the vector form of the delta rule, because training with the 
first stimulus makes w_1 = r. When the second stimulus is presented along 
with the first, its weight starts out at w2 = 0, but the prediction of reward 
v = w_1 u_1 + w_2 u_2 is still equal to r. 

This makes \delta = 0, so no further weight modification occurs.
"""
#Run the blocking trail with the given model.
"""
def run_trail(num_trials, num_stimuli, model, 
                        pre_train_period=(0, .50),
                        train_period=(.50, .75)):
   trail_result = setup_stimuli(num_trials, num_stimuli, pre_train_period=pre_train_period, train_period=train_period)
    
    stimuli = trail_result["stimuli"]
    rewards = trail_result["rewards"]
    
    train_end = int(num_trials * train_period[1]) 
    # update the model with the stimulus and reward, 
    reward_predictions = np.zeros(num_trials) 

    model_predict = model["predict_fn"]
    model_update = model["update_fn"] 
    model_weight_updates = np.zeros((num_stimuli, num_trials)) 

    # Run Model pre-training    
    for i in range(train_end):
        cur_stimulus = stimuli[:, i]
        assert cur_stimulus.shape == (num_stimuli,)
        reward_predictions[i] =  model_predict(model, stimuli[:, i])
        
        # Update model, record model weights. 
        model_update(model, stimuli[:, i], rewards[i], reward_predictions[i])
        model_weight_updates[:, i] = model["weights"]
        
    # run model training 
    for i in range(train_end, num_trials):
        reward_predictions[i] = model_predict(model, stimuli[:, i])
        model_update(model, stimuli[:, i], rewards[i], reward_predictions[i])
        model_weight_updates[:, i] = model["weights"]
    
    # run a single test for all stimuli separately
    test_results = np.zeros(num_stimuli)
    for i in range(num_stimuli):
        one_hot_stimuli = np.zeros(num_stimuli)
        one_hot_stimuli[i] = 1
        test_results[i] = model_predict(model, one_hot_stimuli)

    return {
        "model_weight_updates": model_weight_updates,
        "reward_predictions": reward_predictions, 
        "test_results": test_results,
        "stimuli": stimuli,
        "rewards": rewards, 
        "train_start": trail_result["train_start"],
        "train_end": trail_result["train_end"],
        "pre_train_start": trail_result["pre_train_start"],
        "pre_train_end": trail_result["pre_train_end"]
    }
""" 
            
def setup_stimuli(num_trials, num_stimuli, 
                        pre_train_period=(0, .20), 
                        train_period=(.20, .40), ideal_learning_rate=0.3):
    """
    Blocking Effect:
    
    Pre-Training: s_1 -> r
    Training: s_1 -> r, s_2 -> r
    Result: s_1 -> r, s_2 -> '.' 
    """
    stimuli = np.zeros((num_stimuli, num_trials))
    rewards = np.zeros(num_trials)
    idealized_expected_rewards = np.zeros(num_trials)
   
    # pre-training
    pre_train_start = int(num_trials * pre_train_period[0])
    pre_train_end = int(num_trials * pre_train_period[1])
    pretrain_stimulus_idx = 0

    # s_1 -> r
    prev_expected_reward = 0
    for trial_idx in range(pre_train_start, pre_train_end):
        # During pre-training always show the primary stimulus
        stimuli[pretrain_stimulus_idx, trial_idx] = 1
        rewards[trial_idx] = 1
        
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

    # s_1 + s_2 -> r
    for trial_idx in range(train_start, train_end):
        # During training show both the primary and other stimuli.
        # Present all stimuli
        for stimulus_idx in range(num_stimuli):
            stimuli[stimulus_idx, trial_idx] = 1
            
        # reward stimuli 
        rewards[trial_idx] = 1
        
        print(f"idealized_expected_rewards: {trial_idx} {idealized_expected_rewards[trial_idx-1]}") 
        # Populate previous expected rewards
        idealized_expected_rewards[trial_idx] =  prev_expected_reward
        prev_expected_reward = min(idealized_expected_rewards[trial_idx], 1) 
        
    # TODO: Add idealized test rewards.
    
    # create stimuli, rewards presented in pre-training, 
    # training periods expected rewards in test period.
    return { 
            "stimuli": stimuli, 
            "rewards": rewards, 
            "train_start": train_start,
            "train_end": train_end,
            "pre_train_start": pre_train_start,
            "pre_train_end": pre_train_end,
            "idealized_expected_rewards": idealized_expected_rewards
        }