#%%
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import learning_rule 
import trail_runner 

import blocking_trail
import overshadowing 
import secondary_conditioning
import explaining_away
import inhibitory_conditioning_trail

import importlib

importlib.reload(trail_runner)
importlib.reload(learning_rule)

importlib.reload(blocking_trail)
importlib.reload(overshadowing)
importlib.reload(secondary_conditioning)
importlib.reload(explaining_away)
importlib.reload(inhibitory_conditioning_trail)

IMAGE_ROOT="/Users/aakarsh/src/TUE-WINTER-2024/W24-NEURO-MODELING/hws/hw-2/images"

#%%
#%%
sns.set_theme(style="whitegrid")

#%%
def plot_stimuli(trail_result, plot_title="Trail Results", save_path=None):
    """
    plot_stimuli - Plot the stimuli, rewards and expected rewards
    """
    rewards = trail_result["rewards"]
    
    stimuli = trail_result["stimuli"]
    model_weight_updates = trail_result["model_weight_updates"]
    reward_predictions = trail_result["reward_predictions"]
    
    num_stimuli = stimuli.shape[0]
    num_trials = stimuli.shape[1]
    
    # Create a figure and axis
    fig, ax = plt.subplots(num_stimuli + 3, 1, figsize=(10, 20))
    plt.title(plot_title) 
    
    # Plot the stimuli
    for stimulus_idx in range(num_stimuli):
        ax[stimulus_idx].plot(stimuli[stimulus_idx, :])
        ax[stimulus_idx].set_title(f"Stimulus {stimulus_idx}")
        # vertical line indicating pre-trial, training and test period
        ax[stimulus_idx].axvline(x=trail_result["pre_train_end"], color="r", linestyle="--")
        ax[stimulus_idx].axvline(x=trail_result["train_end"], color="g", linestyle="--")
        ax[stimulus_idx].axvline(x=trail_result["test_results"][stimulus_idx], 
                              color="b", linestyle="--")
    
    # Plot the rewards
    ax[num_stimuli].plot(rewards)
    ax[num_stimuli].set_title("Rewards Given")
    ax[num_stimuli].axvline(x=trail_result["pre_train_end"], color="r", linestyle="--")
    ax[num_stimuli].axvline(x=trail_result["train_end"], color="g", linestyle="--")
    
    # Plot the expected rewards
    ax[num_stimuli + 1].set_title("Reward Expectation/Prediction")
    ax[num_stimuli + 1].plot(reward_predictions, label="Recolra-Wagner Prediction")
    if "idealized_expected_rewards" in trail_result:
        ax[num_stimuli + 1].plot(trail_result["idealized_expected_rewards"], color='g', label='Idealized Expected Rewards')
    ax[num_stimuli + 1].axvline(x=trail_result["pre_train_end"], color="r", linestyle="--")
    ax[num_stimuli + 1].axvline(x=trail_result["train_end"], color="g", linestyle="--")
    # primary reward
    ax[num_stimuli + 1].scatter(y=trail_result["test_results"][0],
                                x=[num_trials]*1, color="r", label="Primary Reward (s1)")
     
    ax[num_stimuli + 1].scatter(y=trail_result["test_results"][1:],
                                x=[num_trials]*(num_stimuli-1), color="b", label="Secondary Rewards (s2)")
    ax[num_stimuli+1].legend()
    
    # Plot the model weight updates
    for stimulus_idx in range(num_stimuli):
        ax[num_stimuli + 2].plot(model_weight_updates[stimulus_idx, :])
        ax[num_stimuli + 2].set_title("Model Weight")
        ax[stimulus_idx].axvline(x=trail_result["pre_train_end"], color="r", linestyle="--")
        ax[stimulus_idx].axvline(x=trail_result["train_end"], color="g", linestyle="--")
    
    plt.legend() 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return fig

#%%
# Blocking Effect: s_1 -> r | s_1, s_2 -> r | s_1 -> r and s_2 -> '.'
num_trials = 100
num_stimuli =  2

def run_blocking_trail(num_trials, num_stimuli, model_under_test):
    pre_train_period = (0, 0.50)
    train_period     = (0.50, 1.0)
    blocking_trail_result = trail_runner.run_trail(num_trials, num_stimuli, 
                                                     model_under_test, 
                                                        blocking_trail.setup_stimuli,
                                                     pre_train_period=pre_train_period,
                                                     train_period=train_period)
    plot_stimuli(blocking_trail_result, save_path=f"{IMAGE_ROOT}/blocking_trail.png") 

run_blocking_trail(num_trials, num_stimuli, learning_rule.rescolra_wagner_create(num_stimuli))

#%% Inhibitory Conditioning: . | s_1 + s_2 -> . and s_1 -> r | s_1 -> r and s_2 -> -r 
num_trials = 100
num_stimuli =  2

def run_inhibitory_conditioning(num_trials, num_stimuli, model_under_test):
    pre_train_period, train_period = (0, 0.50), (0.50, 1.0)
   
    inhibitory_conditioning_result = \
        trail_runner.run_trail(num_trials, num_stimuli,
                                    model_under_test,
                                    inhibitory_conditioning_trail.setup_stimuli,
                                    pre_train_period=pre_train_period,
                                    train_period=train_period)

    plot_stimuli(inhibitory_conditioning_result, save_path=f"{IMAGE_ROOT}/inhibitory_conditioning.png")

run_inhibitory_conditioning(num_trials, num_stimuli,
                                learning_rule.rescolra_wagner_create(num_stimuli))

#%% Overshadowing: . | s_1 + s_2 -> r | s_1 -> \alpha_1 r and s_2 -> \alpha_2 r
num_trials = 100
num_stimuli =  2

def run_overshadowing(num_trials, num_stimuli, model_under_test):
    pre_train_period = (0, 0.50)
    train_period     = (0.50, 1.0)
    train_result = \
        trail_runner.run_trail(num_trials, num_stimuli, 
                               model_under_test, 
                               overshadowing.setup_stimuli, 
                               pre_train_period=pre_train_period, train_period=train_period)
    plot_stimuli(train_result, save_path=f"{IMAGE_ROOT}/overshadowing.png") 


run_overshadowing(num_trials, num_stimuli, 
                    learning_rule.rescolra_wagner_create(num_stimuli))
#%% Secondary Conditioning: 
# s_1 -> r | s_2 -> s_1 | s_2 -> r  
num_trials = 100
num_stimuli =  2

def run_secondary_conditioning(num_trials, num_stimuli, model_under_test):
    pre_train_period = (0, 0.50)
    train_period     = (0.50, 1.0)
    train_result = \
        trail_runner.run_trail(num_trials, num_stimuli, 
                               model_under_test, 
                               secondary_conditioning.setup_stimuli, 
                               pre_train_period=pre_train_period, train_period=train_period)
        
    plot_stimuli(train_result, save_path=f"{IMAGE_ROOT}/secondary_conditioning.png") 

run_secondary_conditioning(num_trials, num_stimuli, 
                                learning_rule.rescolra_wagner_create(num_stimuli))
#%%
# Explaining Away: 
# s_1+s_2 -> r | s_1 -> r | s_1 -> r , s_2 -> '.'
num_trials = 100
num_stimuli =  2

def run_explaining_away(num_trials, num_stimuli, model_under_test):
    pre_train_period = (0, 0.50)
    train_period     = (0.50, 1.0)
    train_result = \
        trail_runner.run_trail(num_trials, num_stimuli, model_under_test, 
                               explaining_away.setup_stimuli, 
                                        pre_train_period=pre_train_period, 
                                        train_period=train_period)
    plot_stimuli(train_result, save_path=f"{IMAGE_ROOT}/explaining_away.png") 

run_explaining_away(num_trials, num_stimuli, 
                        learning_rule.rescolra_wagner_create(num_stimuli))

def run():
    scenarios_fn = [run_blocking_trail, 
                        run_inhibitory_conditioning, 
                        run_overshadowing, 
                        run_secondary_conditioning, 
                        run_explaining_away]
    for scenario_fn in scenarios_fn:
        scenario_fn(num_trials, num_stimuli, 
                        learning_rule.rescolra_wagner_create(num_stimuli))
 
#%%
if __name__ == "__main__":
    #%%
    run()
