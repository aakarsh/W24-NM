#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec

from sortedcontainers import SortedDict
#%%
# We attempt to replicate the figure. 
# z-axis - \delta(t) that is the prediction error.
# x,z - time-steps and number of trials 
# 
# Reward is presented at t = 200
#%%
# Get the directory of the current file uponelevel
current_dir = os.path.dirname(__file__).split("/")[-2] 

# Specify a relative directory for saving figures
output_dir = os.path.join(current_dir, "figures")
os.makedirs(output_dir, exist_ok=True)

IMAGE_PATH = output_dir

#%% Stimulus
num_time_steps = 250

def make_stimulus(n=num_time_steps):
    """."""
    u = np.zeros(n)
    u[100] = 1
    return u
#%%
def add_bump(r, idx, bump_width=3,max_reward=.5, min_reward=0.10):
    """."""
    decrements = np.linspace(max_reward, min_reward, bump_width)
    increments = np.linspace(min_reward, max_reward, bump_width)
    for i, d in enumerate(increments):
        r[idx - bump_width + i] = d
        
    for i, inc in enumerate(decrements):
        r[idx + i] = inc
    return r

#%%
def make_reward(n = num_time_steps):
    """."""
    r = np.zeros(n)
    r = add_bump(r, 200, bump_width=2)
    return r
#%%
def make_reward_by_stimulus_distance(n = num_time_steps, stimulus_position_idx=100, stimulus_distance=50):
    """."""
    r = np.zeros(n)
    r = add_bump(r, stimulus_position_idx + stimulus_distance, bump_width=2)
    return r

#%%
def make_multiple_rewards(n = num_time_steps, num_rewards=2):
    """."""
    bump_idx = 120 
    reward_distance = 50
    r = np.zeros(n)
    for i in range(num_rewards):
        r = add_bump(r, bump_idx + reward_distance * i, bump_width=2)
    return r

#%%
def make_weights(n = num_time_steps):
    """."""
    return np.zeros(n)

def make_value_state(n = num_time_steps):
    """."""
    return np.zeros(n)

def delta(r, v, t):
    """
        \delta(t) = r(t) + v(t + 1) - v(t)
    """
    return r[t] + v[t+1] - v[t] if t < len(v) - 1 else r[t]

def update_weights(w, u, delta, t, epsilon=0.1):
    """
        w(\tau)  = w(\tau) + \epson \delta(t) u(t - \tau)
    """
    for tau in range(len(w)):
        w[tau] = w[tau] + epsilon * delta * u[t - tau]
    return w

def run_trial(model, learning_rate=0.1):
    """
    Run a single trial.
    """
    u, r, v, w = model["stimulus"], model["rewards"], model["values"], model["weights"] 
   
    deltas = np.zeros(len(u)) 
    for t in range(len(u)): # time-step trail 
        d = delta(r, v, t)
        w = update_weights(w, u, d, t, epsilon=learning_rate)
        deltas[t] = d
        # v(t) = \sum_{\tau=0}^{t} w(\tau)  u(t - \tau)
        v[t] = np.dot(w[0:t+1], u[t::-1])
    return v, w, deltas

def train_model(model, num_trials=100, learning_rate=0.1):
    """
    Run multiple trials.
    """
    model = SortedDict(model)
    prediction_errors = [] 
    for _ in range(num_trials): # number of trials
        v, w, deltas = run_trial(model, learning_rate=learning_rate)
        model = update_model(model, v, w, deltas)
        prediction_errors.append(deltas)
    return model, np.array(prediction_errors)

#%%
def pre_train_behavior(model, learning_rate=0.1):
    """
    Pre-train behavior of the model. 
    """
    model = SortedDict(model)
    prediction_errors = []
    v, w, deltas = run_trial(model, learning_rate=learning_rate)
    model = update_model(model, v, w, deltas, update_weights=False)
    prediction_errors.append(deltas)
    return model, np.array(prediction_errors)

#%%
def post_train_behavior(model, learning_rate=0.1):
    """
    Post-train the behavior of the model.
    """
    model = SortedDict(model)
    prediction_errors = []
    v, w, deltas = run_trial(model, learning_rate=learning_rate)
    model = update_model(model, v, w, deltas, update_weights=False)
    prediction_errors.append(deltas)
    return model, np.array(prediction_errors)

def update_model(model, v, w, deltas, update_weights=False):
    """
    Update the model with the new values.
    """
    model = SortedDict(model)
    model["dv"] = np.diff(v)
    model["deltas"] = deltas 
    if update_weights:
        model["values"] = v
        model["weights"] = w
    return model

def initialize_model(num_time_steps=250):
    """
    Initialize the model.
    """
    u = make_stimulus(n=num_time_steps)     # Trial property
    r = make_reward(n=num_time_steps)       # Trial property
    v = make_value_state(n=num_time_steps)  # Model property
    w = make_weights(n=num_time_steps)      # Model property
    dv = np.zeros(num_time_steps)           # Model property

    return SortedDict({ 
             "stimulus": u, 
             "rewards": r, 
             "values": v, 
             "weights": w,
             "dv": dv
    })
#%%
def model_from_template(model, overrides = {}):
    """
    Create a new model from a template.
    """
    return SortedDict({
        "stimulus": np.copy(model["stimulus"]) if "stimulus" not in overrides else overrides["stimulus"],
        "rewards": np.copy(model["rewards"]) if "rewards" not in overrides else overrides["rewards"],
        "values": np.copy(model["values"]) if "values" not in overrides else overrides["values"],
        "weights": np.copy(model["weights"]) if "weights" not in overrides else overrides["weights"],
        "dv": np.copy(model["dv"] if "dv" not in overrides else overrides["dv"])
    })
    
#%%
# Generate example data
def plot_prediction_error(model, deltas, ax=None, save_path=None, delta_min=0, delta_max=2):
    #if ax: ax.set_zlim(delta_min, delta_max)  # Ensure consistent z-axis limits

    u, r, v, w, dv = \
        model["stimulus"], \
        model["rewards"], \
        model["values"], \
        model["weights"], \
        model["dv"]
    # delta : 100(num_trials) x 250 (num_time_steps)
    num_trials, num_time_steps = deltas.shape

    time = np.linspace(0,num_time_steps, num_time_steps)  
    trials = np.linspace(0, num_trials, num_trials)  

    # Mesh Grid for consistent dimensions
    trials, time = np.meshgrid(trials, time) 
    
    assert deltas.shape == trials.T.shape == time.T.shape 
  
    deltas = deltas.T  
    show = False
    if ax is None:
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection='3d')
        show = True

    ax.scatter(0, 0, 0, color='black', s=10, label='O')
 
    surf = ax.plot_surface(time, trials, deltas, 
                           cmap='gray', edgecolor='k', 
                           rstride=45, cstride=45, alpha=0.2)

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Trials')
    ax.set_zlabel('δ(t)')
     
    ax.view_init(elev=15, azim=-110)  
    if show:
        plt.savefig(save_path) if save_path else None
        plt.show()

#%%    
def plot_model_behavior(pre_train_model, pre_train_deltas, 
                        training_model, training_deltas,
                        post_train_model, post_train_deltas, save_path=None):
    
    fig = plt.figure(figsize=(15, 25), constrained_layout=True)
    n_vars = 5 

    gs = GridSpec(n_vars + 1, 2, figure=fig,  height_ratios=[10] + [1] * n_vars)  # Define a grid with 2 rows and 2 columns
    top_graph_ax = fig.add_subplot(gs[0, :], projection='3d')
    

    plot_prediction_error(training_model, training_deltas, ax=top_graph_ax)    

    num_time_steps = pre_train_model["stimulus"].shape[0]
    t = np.linspace(0, num_time_steps, num_time_steps)  # Time points
    # Variables for "before" and "after"
    variables_before = {
        "u": pre_train_model["stimulus"], 
        "r": pre_train_model["rewards"], 
        "v": pre_train_model["values"], 
        "Δv":np.append(np.array([0]), pre_train_model["dv"]),
        "δ": pre_train_deltas.flatten()
    }

    variables_after = {
        "u": post_train_model["stimulus"], 
        "r": post_train_model["rewards"], 
        "v": post_train_model["values"], 
        "Δv":np.append(np.array([0]), post_train_model["dv"]),
        "δ": post_train_deltas.flatten()
    }
   
    axes_before = []
    axes_after = []

    for i in range(n_vars):
        ax_before = fig.add_subplot(gs[i + 1, 0])
        axes_before.append(ax_before)
        ax_after = fig.add_subplot(gs[i + 1, 1])
        axes_after.append(ax_after)
        
    # Plot "before" and "after" for each variable
    for i, (var, data_before) in enumerate(variables_before.items()):
        ax_before =axes_before[i] 

        ax_before.plot(t, data_before, color='black')
        ax_before.axvline(100, color="gray", linestyle="--", linewidth=0.8)  # Key event marker
        if i == 0:
            ax_before.set_title("Before")
        ax_before.set_ylabel(var, rotation=0, labelpad=15)

        # "After" plots
        ax_after = axes_after[i]
        ax_after.plot(t, variables_after[var], color='black')
        ax_after.axvline(100, color="gray", linestyle="--", linewidth=0.8)  # Key event marker
        if i == 0:
            ax_after.set_title("After")

    # Set common labels and layout
    fig.text(0.5, 0.04, 't (time)', ha='center')
    plt.savefig(save_path) if save_path else None
    plt.show()
    
if False:
    #%%
    model = initialize_model()
    #%%
    pre_train_model, pre_train_deltas =\
    pre_train_behavior(model, learning_rate=0.9)
    #%%
    trained_model, train_deltas = \
    train_model(model, num_trials=2000, learning_rate=0.9)
    #%%
    post_train_model, post_train_deltas = \
    post_train_behavior(trained_model, learning_rate=0.9)
    #%%
    plot_model_behavior(pre_train_model, pre_train_deltas, 
                    trained_model, train_deltas,
                    post_train_model, post_train_deltas)
#%%
# TODO don't know why v 
# and dv are non-zero before training
#%%
#%%
## Experiment with following parameters. 
## Plot and briefly describe your observations 
## for each.

#%%
## Reward Timing  (1h)
# 1. Give reward close to stimulus
# 2. Give reward far from stimulus
def experiment_reward_timing(selected_condition, num_trials=2000, learning_rate=0.5):
    model = initialize_model()
    if selected_condition == "NEAR_STIMULUS":
        model = model_from_template(model, overrides={"rewards": make_reward_by_stimulus_distance(stimulus_distance=20)})
    elif selected_condition == "FAR_STIMULUS":
        model = model_from_template(model, overrides={"rewards": make_reward_by_stimulus_distance(stimulus_distance=140)})
    else:
        raise ValueError("Invalid condition") 
    
    pre_train_model, pre_deltas = pre_train_behavior(model, learning_rate=learning_rate)
    trained_model, train_deltas = train_model(model, num_trials=num_trials, learning_rate=learning_rate)
    post_train_model, post_train_deltas = post_train_behavior(trained_model, num_trials=num_trials, learning_rate=learning_rate)

    save_path = os.path.join(IMAGE_PATH, f"experiment_reward_timing_{selected_condition}.png")

    plot_model_behavior(pre_train_model, pre_deltas,
                        trained_model, train_deltas, 
                        post_train_model, post_train_deltas, 
                        save_path=save_path) 

for condition in ["NEAR_STIMULUS", "FAR_STIMULUS"]:
    experiment_reward_timing(condition)
#%%
## Learning Rate (1h)
# 1. High Learning Rate
# 2. Low Learning Rate
def experiment_learning_rate(selected_condition, num_trials=2000, high_lr=0.5, low_lr=0.05):
    learning_rate = high_lr if "HIGH_LR" == selected_condition else low_lr 
    model = initialize_model()
    pre_train_model, pre_deltas = pre_train_behavior(model, learning_rate=learning_rate)
    trained_model, train_deltas = train_model(model, num_trials=num_trials, learning_rate=learning_rate)
    post_train_model, post_train_deltas = post_train_behavior(trained_model, num_trials=num_trials, learning_rate=learning_rate)

    save_path = os.path.join(IMAGE_PATH, f"experiment_learning_rate_{selected_condition}.png")

    plot_model_behavior(pre_train_model, pre_deltas,
                        trained_model, train_deltas, 
                        post_train_model, post_train_deltas, 
                        save_path=save_path)

for condition in ["HIGH_LR", "LOW_LR"]:
    experiment_learning_rate(condition) 
    
#%%
## Multiple Rewards (3h)
# 1. Provide two rewards.
# 2. Provide three rewards.
def experiment_multiple_rewards(selected_condition, num_trials=2000, learning_rate=0.5, num_rewards=2):
    model = initialize_model()
    if selected_condition == "SINGLE_REWARD":
        model = model_from_template(model, overrides={"rewards": make_reward()})
    elif selected_condition == "MULTIPLE_REWARDS":
        model = model_from_template(model, overrides={"rewards": make_multiple_rewards(num_rewards=num_rewards)})
    else:
        raise ValueError("Invalid condition") 
    
    pre_train_model, pre_deltas = pre_train_behavior(model, learning_rate=learning_rate)
    trained_model, train_deltas = train_model(model, num_trials=num_trials, learning_rate=learning_rate)
    post_train_model, post_train_deltas = post_train_behavior(trained_model, num_trials=num_trials, learning_rate=learning_rate)

    save_path = os.path.join(IMAGE_PATH, f"experiment_multiple_rewards_{selected_condition}.png")

    plot_model_behavior(pre_train_model, pre_deltas,
                        trained_model, train_deltas, 
                        post_train_model, post_train_deltas, 
                        save_path=save_path)

for condition in ["SINGLE_REWARD", "MULTIPLE_REWARDS"]:
    experiment_multiple_rewards(condition) 
 

#%%
## Stochastic Rewards (4h)
# 1. Randomly provide rewards with low noise
# 2. Randomly provide rewards with high noise
