#%%
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sortedcontainers import SortedDict
#%%
"""
Temporal Difference Learning:

    t: time step t \in {0, 1, 2, ..., T}
    u(t): \in \mathbb{R}^d stimulus at time t.
    v(t): \in \mathbb{R}^d prediction at time t.
    r(t): \in \mathbb{R}^d reward at time t.
    
    v(t) is interpreted as total expected reward from time t  
    to the end of the episode T. Of course the animal 
    does not know when the episode ends. 
   
   v(t)  = \sum_{\tau=0}^{T - t} r(t + \tau)
   
   or equivalently informally:
   v(t) = r(t) + r(t + 1) + r(t + 2) + ... + r(T)
   
   where T is the end of the episode.
   
   The animal is assumed to average over multiple trials 
   and it is denoted as:
   
   v(t) = \langle \sum_{\tau=0}^{T - t} r(t + \tau) \rangle
   
   The generalized approximation  of v(t) is:
   
   v(t)= \sum_{\tau=0}^{t} w(\tau)  u(t - \tau)
   
  Modified delta rule:
    w(\tau)  = w(\tau) + \epson \delta(t) u(t - \tau) 
    
 where:
    \delta(t)  = \sum_{\tau} r( t + \tau ) - v(t)

    computed-rewards - predicted-rewards

We create a recusive fomulation usign future rewards
\sum_{\tau=0}^{T - t} r(t + \tau) = r(t) + \sum_{\tau=0}^{T - t - 1} r(t + 1 + \tau)


But it is assumed that v(t) provides a good approximation of the future rewards.
That is 

\sum_{\tau=0}^{T - t} r(t + \tau) \approx  r(t) + v(t + 1)

This approximation is used to compute the temporal difference error:

\delta(t) = r(t) + v(t + 1) - v(t)
and 
    w(\tau)  = w(\tau) + \epson \delta(t) u(t - \tau)
"""
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
    """
    .
    """
    u = np.zeros(n)
    u[100] = 1
    return u

def make_reward(n = num_time_steps):
    """
    .
    """
    r = np.zeros(n)
    r[200] = 1
    return r

def make_weights(n = num_time_steps):
    """
    .
    """
    return np.zeros(n)

def make_value_state(n = num_time_steps):
    """
    """
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

def run_trial(model):
    """
    Run a single trial.
    """
    u, r, v, w = model["stimulus"], model["rewards"], model["values"], model["weights"] 
    
    for t in range(len(u)): # time-step trail 
        d = delta(r, v, t)
        w = update_weights(w, u, d, t)
        # v(t) = \sum_{\tau=0}^{t} w(\tau)  u(t - \tau)
        v[t] = np.dot(w[0:t+1], u[t::-1])
    return v, w

def train_model(model, num_trials=100):
    """
    Run multiple trials.
    """
    for _ in range(num_trials):
        v, w = run_trial(model)
        model = update_model(model, v, w)
    return model 

def update_model(model, v, w):
    """
    Update the model with the new values.
    """
    model = SortedDict(model)
    model["values"] = v
    model["weights"] = w
    model["dv"] = np.diff(v)
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

def plot_final_state(model):
    u, r, v, w, dv = model["stimulus"], model["rewards"], model["values"], model["weights"], model["dv"]
    sns.set_theme()
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    plt.plot(u)
    plt.title("Stimulus")
    plt.subplot(2, 2, 2)
    plt.plot(r)
    plt.title("Reward")
    plt.subplot(2, 2, 3)
    plt.plot(v)
    plt.title("Value State")
    plt.subplot(2, 2, 4)
    plt.plot(dv)
    plt.title("\Delta v ")
    plt.tight_layout()
    plt.show()
    
#%%
model = initialize_model()
model = train_model(model, num_trials=1)
# - somehting.
plot_final_state(model)
#%%
