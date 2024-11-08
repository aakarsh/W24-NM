import numpy as np

# num_trials, num_stimuli, 
# pre_train_period=(0, .50), train_period=(.50, .75)
def run_trail(num_trials, num_stimuli, model,
                        setup_stimuli,  
                        pre_train_period=(0, .50),
                        train_period=(.50, .75)):
    """
    Run a trail with the given model and 
    stimuli setup.
    """
    trail_result = setup_stimuli(num_trials, 
                                 num_stimuli, 
                                 pre_train_period=pre_train_period, 
                                 train_period=train_period)
    
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
        "idealized_expected_rewards": trail_result["idealized_expected_rewards"],
        "train_start": trail_result["train_start"],
        "train_end": trail_result["train_end"],
        "pre_train_start": trail_result["pre_train_start"],
        "pre_train_end": trail_result["pre_train_end"]
    }