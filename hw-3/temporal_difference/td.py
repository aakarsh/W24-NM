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
"""
- rewards must be a bump
- if you know the states
- you only need to multiply your reward function with your future visitation of the states.
- successor representation it always goes right, we now came about 
the successor representation fo $s_1$ and policy is always go to like this. 
- the animal does this thinga bunch of times. 
- Then what my successor representation is going to look like. 
- We start ins $s_1$ and i am here now. 
- if i am here and i will visit state 1   
- how often will i visit state 2. 
- at least once as well. 
- two you can reach from 3 and 1
- it is not necessarily the maximum 
- we are talking about the discounted future visitation of the states.
- some valeu close to 1. 
- We will visit state 2, we will count it to 0.98 offset. 
- What will happen for state3. 
- What will be the height of this. around 0.96 \gamma^2 
- We will keep declining hwo many states 
- how often will we visit state 6 0 times.
- successor representation is very policy dependent. 
- This is going to determine how it 
- The expected sum of future discounted states. 
- we will add a gamma^2 to state3 but then wi will visit again gamma^3 
- wo sit will be some
-  so keep adding thims up when we visit up.
- so rest is going to look all the same with another tow gammas because 
- we visited it a bit later  
- This is what sr does - how often we will visit 

- How to learn this ? 
- In TD you comapred current with rewards you boserve, and compute the td between them. 
- for the sr what you do is you do it in same way. 
- and things become easier, because in TD we need to keep track of time, 
- but sr has allthe ftuure kind of collapesed into one. 

- The way to update is to do a trajectory, keep count of which state and when. 
- then a vector of when you visited them , 
- compute the prediction error and 
- its intutive to update this. 
- you do need to collect whole sequence of visitations. 
- i go up to the maze , need to collect th whole trajectory then 
- you can update the SR. 
- talk a bit when you have tried it. 
- So state occupancy has to do with reward ??
- We have all the future state visitations collapse 
- into representation.

- We can combine our future state occupancy with our rewards. 

- we have T maze and we go randomly this way or that way. 
- They will be a bit same and we might go left or right. 
- so we say there will be reward in state 9. 
- compute the dot product of future state occupancy with your .... 
   


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
    """."""
    u = np.zeros(n)
    u[100] = 1
    return u
#%%
def add_bump(r, idx, bump_width=5):
    """."""
    decrements = np.linspace(1, 0.15, bump_width)
    increments = np.linspace(0.15, 1, bump_width)
    for i, d in enumerate(increments):
        r[idx - bump_width + i] = d
        
    for i, inc in enumerate(decrements):
        r[idx + i] = inc
    return r

#%%
def make_reward(n = num_time_steps):
    """."""
    r = np.zeros(n)
    r = add_bump(r, 200, bump_width=10)
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
    print(f"num trails: {num_trials}")
    print(f"timestapm:deltas[0].shape: {deltas.shape}")
    return model, np.array(prediction_errors)

def update_model(model, v, w, deltas):
    """
    Update the model with the new values.
    """
    model = SortedDict(model)
    model["values"] = v
    model["weights"] = w
    model["dv"] = np.diff(v)
    model["deltas"] = deltas 
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
def plot_prediction_error(model, deltas):
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

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(0, 0, 0, color='black', s=10, label='O')
 
    surf = ax.plot_surface(time, trials, deltas, 
                           cmap='gray', edgecolor='k', 
                           rstride=70, cstride=70, alpha=0.3)
    # Add color bar to show the mapping of colors to z-values
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Trials')
    ax.set_zlabel('δ(t)')
     
    ax.view_init(elev=15, azim=-100)  
    plt.show()


#%%    
def plot_final_state(model):
    u, r, v, w, dv = \
        model["stimulus"], \
        model["rewards"], \
        model["values"], \
        model["weights"], \
        model["dv"]
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
model, deltas = train_model(model, num_trials=2000, 
                            learning_rate=0.9)
#%%
plot_final_state(model)
#%%
plot_prediction_error(model, deltas)    
#%%
## Experiment with following parameters. 
## Plot and briefly describe your observations 
## for each.

#%%
## Reward Timing 
#%%
## Learning Rate
#%%
## Multiple Trials
#%%
## Stochastic Rewards
#%% 

# %%
