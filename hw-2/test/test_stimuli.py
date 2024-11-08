import stimulus.blocking_trail as blocking_trail
import stimulus.learning_rule as learning_rule
import numpy as np

np.random.seed(0)

# inhibitory-conditioning
# over-shadowing
# secondary-conditioning
# explaining-away

        
# blocking
def test_zero_model_blocking_trail():
    num_trials = 100
    num_stimuli = 2
    
    model_under_test = learning_rule.zero_model_create()

    reward_predictions, expected_results = \
        blocking_trail.run_trail(num_trials, num_stimuli, model_under_test)

    print("reward-predictions:", reward_predictions)
    print("expected-results:", expected_results)

    assert np.allclose(reward_predictions, np.zeros(len(reward_predictions))) 
    
def test_rescolra_wagner_blocking_trail():
    """
    """
    num_trials  = 100
    num_stimuli = 2
    
    model_under_test = \
        learning_rule.rescolra_wagner_create()

    reward_predictions, expected_results = \
        blocking_trail.run_trail(num_trials, num_stimuli, model_under_test)
    
    test_reward_predictions = \
        reward_predictions[ len(reward_predictions) - len(expected_results) : ] 
   
    assert expected_results.shape == test_reward_predictions.shape
    assert not np.allclose(test_reward_predictions, expected_results)
   
def test_resolca_wagner_blocking_trail_simple():
    np.random.seed(0)
    num_trials = 10
    num_stimuli = 2
    model_under_test = learning_rule.rescolra_wagner_create()
    trail_result = blocking_trail.run_trail(num_trials, num_stimuli, 
                                            model_under_test)
    
    rewards = trail_result["rewards"]
    expected_rewards = trail_result["expected_rewards"]
    stimuli = trail_result["stimuli"]
    model_weights = trail_result["model_weight_updates"]

    print("stimuli:", stimuli)
    stimuli_expected = np.array([[0., 1., 1., 0., 1.,  1.,  1.,  1.,  1.,  1.],
                                 [0., 0., 0., 0., 0.,  1.,  1.,  1.,  1.,  1.]])
    assert np.allclose(stimuli, stimuli_expected)
    
    print("rewards:", rewards)
    rewards_expected = np.array([[0., 1., 1., 0., 1., 1., 1., 0., 0., 0.]])
    assert np.allclose(rewards, rewards_expected)
    
    pre_train_end = trail_result["pre_train_end"]
    for i in range(pre_train_end):
        print(f"pretrain_period [0, {pre_train_end}] (i: {i}): stimulus",
              stimuli[:, i], "rewards", rewards[i])
        #print(f"pretrain_period [0, {pre_train_end}] (i: {i}): expected_rewards", expected_rewards[i])
        print(f"pretrain_period [0, {pre_train_end}] (i: {i}): model_weights", 
            model_weights[: , i])
    expected_pretrain_weights = np.array([
        [0., 0.],
        [0.1, 0.],
        [0.19, 0.],
        [0.19, 0.],
        [0.271, 0.]
    ]) 
    assert np.allclose(model_weights[:, :pre_train_end].T, expected_pretrain_weights)
   
    train_start = trail_result["train_start"]
    train_end = trail_result["train_end"]
    for i in range(train_start, train_end):
        print(f"train_period [{train_start}, {train_end}] (i: {i}): stimulus",
              stimuli[:, i], "rewards", rewards[i])
        #print(f"train_period [{train_start}, {train_end}] (i: {i}): expected_rewards", expected_rewards[i])
        print(f"train_period [{train_start}, {train_end}] (i: {i}): model_weights", 
            model_weights[: , i]) 
        
    test_start = trail_result["test_start"]
    test_end = trail_result["test_end"]
    for i in range(test_start, test_end):
        print(f"test_period [{test_start}, {test_end}] (i: {i}): stimulus",
              stimuli[:, i], "rewards", rewards[i])
        #print(f"test_period [{test_start}, {test_end}] (i: {i}): expected_rewards", expected_rewards[i])
        print(f"test_period [{test_start}, {test_end}] (i: {i}): model_weights", 
            model_weights[: , i])
    
    print("model_weights:", model_weights)
    print("expected-rewards:", expected_rewards)