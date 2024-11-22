#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sortedcontainers import SortedDict
#%%
#%%
# We attempt to replicate the figure. 
# z-axis - \delta(t) that is the prediction error.
# x,z - time-steps and number of trials 
# 
# Reward is presented at t = 200
#
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
    prediction_errors = [] 
    for _ in range(num_trials): # number of trials
        v, w, deltas = run_trial(model)
        model = update_model(model, v, w, deltas)
        prediction_errors.append(deltas)
    return model, np.array(prediction_errors)
#%%
def pre_train_behavior(model, learning_rate=0.1):
    """
    Pre-train behavior of the model. 
    """
    prediction_errors = []
    v, w, deltas = run_trial(model, learning_rate=learning_rate)
    model = update_model(model, v, w, deltas, update_weights=False)
    prediction_errors.append(deltas)
    return model, np.array(prediction_errors) 

#%%
def post_train_behavior(model, num_trials=100, learning_rate=0.1):
    """
    Post-train the behavior of the model.
    """
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
# Generate example data
def plot_prediction_error(model, deltas, save_path=None):
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

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(0, 0, 0, color='black', s=10, label='O')
 
    surf = ax.plot_surface(time, trials, deltas, 
                           cmap='gray', edgecolor='k', 
                           rstride=70, cstride=70, alpha=0.3)

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Trials')
    ax.set_zlabel('δ(t)')
     
    ax.view_init(elev=15, azim=-110)  
    plt.show()
    plt.savefig(save_path) if save_path else None

#%%    
def plot_model_behavior(pre_train_model, pre_train_deltas, 
                        post_train_model, post_train_deltas, save_path=None):

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

    # Create the figure for Panel B
    n_vars = len(variables_before)
    fig, axes = plt.subplots(n_vars, 2, figsize=(14, 8), sharex=True, sharey=True)

    # Plot "before" and "after" for each variable
    for i, (var, data_before) in enumerate(variables_before.items()):
        # "Before" plots
        ax_before = axes[i, 0]
        ax_before.plot(t, data_before, color='black')
        ax_before.axvline(100, color="gray", linestyle="--", linewidth=0.8)  # Key event marker
        if i == 0:
            ax_before.set_title("Before")
        ax_before.set_ylabel(var, rotation=0, labelpad=15)

        # "After" plots
        ax_after = axes[i, 1]
        ax_after.plot(t, variables_after[var], color='black')
        ax_after.axvline(100, color="gray", linestyle="--", linewidth=0.8)  # Key event marker
        if i == 0:
            ax_after.set_title("After")

    # Set common labels and layout
    fig.text(0.5, 0.04, 't (time)', ha='center')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path) if save_path else None
    

#%%
model = initialize_model()
#%%
pre_train_model, pre_deltas = pre_train_behavior(model, learning_rate=0.9)
#%%
trained_model, train_deltas = train_model(model, num_trials=2000, learning_rate=0.9)
#%%
post_train_model, post_train_deltas = post_train_behavior(trained_model, num_trials=2000, learning_rate=0.9)
#%%
plot_model_behavior(pre_train_model, pre_deltas, post_train_model, post_train_deltas)
#%%
#TODO don't know why v and dv are non-zero before training
#%%
plot_prediction_error(model, train_deltas)    
#%%
## Experiment with following parameters. 
## Plot and briefly describe your observations 
## for each.

#%%
## Reward Timing  (1h)
# 1. Give reward close to stimulus
# 2. Give reward far from stimulus
def experiment_reward_timing(stimulus_distance=20):
    model = initialize_model()
    model["rewards"] = add_bump(model["rewards"], 200 + stimulus_distance, bump_width=2)
    pre_train_model, pre_deltas = pre_train_behavior(model, learning_rate=0.9)
    trained_model, train_deltas = train_model(model, num_trials=2000, learning_rate=0.9)
    post_train_model, post_train_deltas = post_train_behavior(trained_model, num_trials=2000, learning_rate=0.9)
    plot_model_behavior(pre_train_model, pre_deltas, post_train_model, post_train_deltas)
    plot_prediction_error(model, train_deltas)
    
#%%
## Learning Rate (1h)
# 1. High Learning Rate
# 2. Low Learning Rate

#%%
## Multiple Rewards (3h)
# 1. Provide two rewards
# 2. Provide three rewards 

#%%
## Stochastic Rewards (4h)
# 1. Randomly provide rewards with low noise
# 2. Randomly provide rewards with high noise
#%% 
