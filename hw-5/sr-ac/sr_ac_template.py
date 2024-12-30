#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from datetime import datetime
import os
import numba
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
from scipy.ndimage import gaussian_filter

import importlib
import maze_utils
importlib.reload(maze_utils)

from maze_utils import (
                        update_sr_after_episode, 
                        make_maze, 
                        plot_maze, 
                        plot_trajectory, 
                        plot_stepwise_v_weights_history, 
                        plot_v_weights, 
                        plot_sr_history)
sns.set_theme()
#%%
# Get the absolute path of the current file
absolute_path = os.path.abspath(__file__)
# Get the directory of the current file, one level up
current_dir = os.path.dirname(absolute_path)  # Current file's directory
parent_dir = os.path.dirname(current_dir)    # One level up
# Specify a relative directory for saving figures
output_dir = os.path.join(current_dir, "images")

os.makedirs(output_dir, exist_ok=True)
IMAGE_PATH = output_dir
print(f"Using image path: {IMAGE_PATH}")
#%%
# Define maze
maze = np.zeros((9, 13))

#%%
# Place walls
maze[2, 6:10] = 1
maze[-3, 6:10] = 1
maze[2:-3, 6] = 1

# Define start
start = (5, 7)

#%%
# Define goal (we abuse function scoping a bit here, later we will change the goal, 
# which will automatically change the goal in our actor critic as well)
goal = (1, 1)
goal_state = goal[0]*maze.shape[1] + goal[1]
goal_value = 10

#%%
def plot_maze(maze):
    plt.imshow(maze, cmap='binary')

    # draw thin grid
    for i in range(maze.shape[0]):
        plt.plot([-0.5, maze.shape[1]-0.5], [i-0.5, i-0.5], c='gray', lw=0.5)
    for i in range(maze.shape[1]):
        plt.plot([i-0.5, i-0.5], [-0.5, maze.shape[0]-0.5], c='gray', lw=0.5)

    plt.xticks([])
    plt.yticks([])

#%%
plot_maze(maze)

#%%
def compute_transition_matrix(maze):
    # for a given maze, compute the transition matrix from any state to any other state under a random walk policy
    # (you will need to think of a good way to map any 2D grid coordinates onto a single number for this)

    # create a matrix over all state pairs
    transitions = np.zeros((maze.size, maze.size))

    # iterate over all states, filling in the transition probabilities 
    # to all other states on the next step (only one step into the future)
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            # check if state is valid
            if maze[i, j] == 0:
                # iterate over all possible moves
                for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_i, new_j = i + move[0], j + move[1]
                    # check if new state is valid
                    if new_i >= 0 and new_i < maze.shape[0] and new_j >= 0 and new_j < maze.shape[1] and maze[new_i, new_j] == 0:
                        transitions[i*maze.shape[1] + j, new_i*maze.shape[1] + new_j] = 1
    
    # normalize transitions
    transitions /= transitions.sum(axis=1, keepdims=True)

    # remove NaNs
    transitions[np.isnan(transitions)] = 0

    return transitions

transitions = compute_transition_matrix(maze)


#%%
def random_walk_sr(transitions, gamma):
    return np.linalg.inv(np.eye(transitions.shape[0]) - gamma * transitions.T)

def analytical_sr(transitions, gamma):
    return np.linalg.inv(np.eye(transitions.shape[0]) - gamma * transitions.T)

#%%
i, j = start
# compute the SR for all states, based on the transition matrix
# note that we use a lower discounting here, to keep the SR more local
analytical_sr = analytical_sr(transitions, 0.8).T
analytical_sr_read_only = analytical_sr[:]
#%%
# Part 1: Program an actor-critic algorithm:
# To navigate the maze,  
# Actor:
#   Using a table of action propensities M with softmax action selection
# Critic:
#   a Learned state-value function as critic
#%% 
@numba.jit
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / (np.sum(exp_x) + 1e-12)

@numba.jit
def position_idx(i, j, maze):
    return i * maze.shape[1] + j

@numba.jit
def position_from_idx(position_idx, maze):
    return position_idx // maze.shape[1], position_idx % maze.shape[1]

#%%
@numba.jit
def normal_start():
    # suggested encoding of 2D location onto states
    i, j = start
    state =  position_idx(i,j, maze) 
    return state

@numba.jit
def is_inside_maze(maze, move):
    return move[0] >= 0 and move[0] < maze.shape[0] and move[1] >= 0 and move[1] < maze.shape[1]

@numba.jit
def is_free_cell(maze, move):
    return maze[move[0], move[1]] == 0

@numba.jit
def check_legal(maze, move):
    return is_inside_maze(maze, move) and is_free_cell(maze, move)

@numba.jit
def legal_move_names():
    return ['up', 'down', 'right', 'left']

@numba.jit
def legal_moves():
    return [(0, 1), (0, -1), (1, 0), (-1, 0)]

@numba.jit
def possible_moves(maze, pos):
    return [(pos[0] + move[0], pos[1] + move[1]) for move in legal_moves()]

@numba.jit
def reachable_moves(maze, pos):
    # return a list of all reachable moves from a given position
    possible_moves = [(pos[0] + move[0], pos[1] + move[1]) for move in legal_moves()]
    legal_moves = [move for move in possible_moves if check_legal(maze, move)]
    return legal_moves 

@numba.jit
def init_propensities(maze, epsilon = 1e-6):
    M =  np.ones((maze.size, 4))* (-1e10)
    random_initialization = np.random.random(M.shape) * epsilon
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            next_moves = possible_moves(maze, (i, j)) 
            reachable_moves = [(move, action) for action, move in enumerate(next_moves) if check_legal(maze, move)]
            for _, action in reachable_moves: 
                M[position_idx(i, j, maze), action] = random_initialization[i, j]
    return M

#%%
"""
Learning to act

We will program an actor-critic to learn a policy on the maze world. 

- Actor
    - Our actor  will be a direct actor. 
    - M - It will learn a table M of action propensities, 
        for each state \times action pair.
        - These action propensities become actions through a softmax function.
- Critic
    - instantiating the value functio V
    - It will be the product of state representation 
        $X$ and learned weights $w$
        $V(s) = X(s) \cdot w$ 
        for any state $s$
- Reward Position (1,1)
"""
#%%
def actor_critic(state_representation, 
                    n_steps, 
                    alpha, 
                    gamma, 
                    n_episodes,
                    sr_regularization=0.01, 
                    update_sr=False, 
                    start_func=normal_start, 
                    v_init=None,
                    goal_reach_reward=goal_value, 
                    step_penalty=0,
                    goal=goal, 
                    clip_weight_max_value=1e5,
                    enable_performance_counters=False,
                    debug=False):
    # Implement the actor-critic algorithm to learn to navigate the maze
    #
    # state_representation - 
    #       $sr \in \mathbb{R}^{n_states \times n_states}$, 
    #       Giving us the representation for each, which is either a 1-hot 
    #       vector (so e.g. state_representation[15] is a vector of size n_states 
    #       which is 0 everywhere, except 1 at index 15), or the SR for each state
    #
    # n_steps - is the number of actions in each episode before 
    # it gets cut off, an episode also ends when the agent 
    # reaches the goal.
    # 
    # alpha - learning rate
    # gamma - discount factor 
    # n_episodes - is the number of episodes to train the agent
    # 
    # update_sr - is for exercise part 3, 
    #               when you want to update the SR, after each episode
    #               start_func allows you to specify a different starting 
    #               state, if desired
    # Initialize M-table

    perf_counters = {
                     "num_episodes": 0, 
                     "num_steps": 0, 
                     "num_goal_reached": 0,
                     'num_clipped_weights': 0, 
                     'SR_history': None,
                     'M_history': None,
                     'M_history_step_wise': None, 
                     'V_weight_history': None,
                     'V_weight_history_step_wise': None,
                     "locals" : {
                         'sr_regularization': sr_regularization,
                         "n_steps": n_steps, 
                         "alpha": alpha, 
                         "gamma": gamma, 
                         "n_episodes": n_episodes,
                         "update_sr": update_sr,
                         "goal_reach_reward": goal_reach_reward,
                         "step_penalty": step_penalty,
                         "goal": goal,
                         "clip_weight_max_value": clip_weight_max_value
                    } 
    }

    tracked_states = [position_idx(i, j, maze) for i in range(maze.shape[0]) for j in range(maze.shape[1]) if check_legal(maze, (i, j))]
    M = init_propensities(maze)
    # Initialize state-value function
    num_states = state_representation.shape[0]
    # w - weights for the value function
    V_weights = np.zeros(num_states) if v_init is None else v_init 
    episode_rewards = np.zeros(n_episodes) # TODO

    LEGAL_MOVES = legal_moves()

    episode_counters = [] 
    # V_weights history are growing unctrollably
    # need to look at SR values as 2d graph as well.
    V_weight_history = np.zeros((n_episodes, num_states))
    V_weight_history_step_wise = np.zeros((n_episodes, n_steps, num_states))
    SR_history = np.zeros((n_episodes, num_states, num_states))
    M_history = np.zeros((n_episodes, num_states, 4))
    M_history_step_wise = np.zeros((n_episodes, n_steps, num_states, 4))

    # Iterate over episodes
    for episode_idx in range(n_episodes):
        # Initializations
        # Move to the start state/possibly random start state
        per_episode_counters = {} 
        if enable_performance_counters:
            per_episode_counters["state_visit_counts"] = np.zeros(num_states)
            episode_counters.append(per_episode_counters)
            perf_counters["num_episodes"] += 1
        
        state_idx = start_func()
        
        if debug and episode_idx % 100 == 0: 
            print(f"Episode: {episode_idx}  Starting: {position_from_idx(state_idx, maze)}") 
        
        # Cumulative discount factor
        I = 1
        # episode trajectory
        trajectory = []
        goal_reached=False
        # Go until goal is reached
        for step_idx in range(n_steps):
            if step_idx % 25 == 0 and debug: 
                print(f"Episode: {episode_idx} Step: {step_idx} - Position: {position_from_idx(state_idx, maze)}")
                plot_stepwise_v_weights_history(V_weight_history_step_wise, episode_idx, step_idx) 
            if enable_performance_counters:
                perf_counters["num_steps"] += 1
            # Act and Learn (Update both M and V_weights)
          
            is_legal_move = False
            move= None
            new_state = None
            max_checks = 100 
            checks = 0
            action_probabilities = None
            valid_probabilities = None
            valid_actions = None
            chosen_action = None
            
            while not is_legal_move and checks < max_checks:
                checks += 1
                # Compute Action probabilities - How to only consider valid actions
                valid_actions = [ action for action, move in enumerate(LEGAL_MOVES) 
                                    if check_legal(maze, tuple(np.array(position_from_idx(state_idx, maze)) + np.array(move))) ]

                valid_propensities = np.array([M[state_idx, action] if action in valid_actions else -np.inf 
                                               for action, _ in enumerate(LEGAL_MOVES)])

                action_probabilities = softmax(valid_propensities) 

                if np.isnan(action_probabilities).any():
                    action_probabilities =  np.array([ 
                        1/len(valid_actions) 
                            if action in valid_actions else 0 for action in range(len(LEGAL_MOVES))
                    ])
                    chosen_action = np.random.choice(valid_actions)
                else: 
                    chosen_action = np.random.choice(
                        len(action_probabilities), p=action_probabilities)

                move = LEGAL_MOVES[chosen_action]
                new_state = tuple(np.array(position_from_idx(state_idx, maze)) + np.array(move))
                i, j = new_state
                new_state_idx = position_idx(i, j, maze)
                is_legal_move = check_legal(maze, new_state)
                if not is_legal_move and enable_performance_counters:
                    per_episode_counters["rejections"] = per_episode_counters.get("rejections", 0) + 1
                    if debug: 
                        print(f"Rejected move: ({i}, {j})")
                elif is_legal_move and enable_performance_counters:
                    per_episode_counters["actions"] = per_episode_counters.get("actions", 0) + 1
                    
            # Should have found legal move
            if not is_legal_move and checks == max_checks: 
                print("valid_actions:", valid_actions)
                print("valid_probabilities:", valid_probabilities)
                print("action_probabilities:", action_probabilities)
                print("M[state_idx, :]:", M[state_idx, :])
                print("state representation:", state_representation[state_idx])
                print("v_weights:", V_weights)
                raise ValueError(f"Could not find a legal move after {max_checks} checks, No legal move for state: {position_from_idx(state_idx, maze)}")
           
            if enable_performance_counters: 
                per_episode_counters["state_visit_counts"][state_idx] += 1 
            trajectory.append(new_state_idx)
            
            V_state = V_weights @ state_representation[state_idx]
            goal_reached = new_state == goal   
            # Compute the value of the new state, goal-state has value 0 
            # V(s) = X(s) \cdot w || V(s) = 0 if s is goal
            V_new_state = V_weights @ state_representation[new_state_idx]  \
                if not goal_reached else 0
            # 
            V_diff = ( gamma * V_new_state ) - V_state 
            reward = goal_reach_reward if goal_reached else step_penalty
            # TD error 
            delta = reward + V_diff
            # Linear function \nabla_{V_weights} X(s)* V_weights  = X(s)
            V_weights += alpha * delta * state_representation[state_idx]

            if np.max(np.abs(V_weights)) > clip_weight_max_value:
                if enable_performance_counters:
                    perf_counters["num_clipped_weights"] += len(V_weights[np.where(np.abs(V_weights) > clip_weight_max_value)])
                # print(f"WARNING: V_weights: {num_overflow_weights} overflowing weights clipped weights") 
                V_weights = np.clip(V_weights, -clip_weight_max_value, clip_weight_max_value)
                 
            # Save history
            V_weight_history[episode_idx, :] = V_weights
            V_weight_history_step_wise[episode_idx, step_idx, :] = V_weights
            # Assuming same \alpha^\theta == \alpha^\theat) :( ?
            # Reduce the probability of the not-chosen action 
            M[state_idx, :] += alpha * I * delta * (-action_probabilities) 
            M[state_idx, chosen_action] += alpha * I * delta * (1) # so we have net (1 - action_probabilities[chosen_action]) increase in probability
            M_history_step_wise[episode_idx, step_idx, :, :] = M
           
            # Absorbing state  
            if (i, j) == goal: 
                episode_rewards[episode_idx] = I * reward # earned rewards for this episode
                break # END EPISODE
            
            state_idx = new_state_idx 
            I *= gamma

        # Episode ended due to max steps 
        if step_idx == n_steps - 1 and not goal_reached:
            episode_rewards[episode_idx] = 0

        if enable_performance_counters:
            per_episode_counters["trajectory"] = np.copy(np.array(trajectory))
            # Save the M history
            M_history[episode_idx, :, :] = M
            # Save step wise M history
            
        # Update the state representation 
        if update_sr and len(trajectory) > 0: 
            state_representation = np.copy(update_sr_after_episode(state_representation, trajectory, gamma, alpha, regularization=sr_regularization))
            # Save the state representation history
            SR_history[episode_idx, :, :] = np.copy(state_representation)
            
            assert (state_representation >= 0).all(), "state representation should be positive" 
            assert (state_representation <= 1/(1-gamma)).all(), "state representation should be less than 1"

    # Reward only for reaching the goal, thus episode reward is 
    # same as Discounted last step reward.
    if enable_performance_counters:
        perf_counters["episode_counters"] = episode_counters
         
    if debug and enable_performance_counters: 
        print(perf_counters)
        
    # create pickle file name with current timestamp 
    if enable_performance_counters:
        perf_counters["V_weight_history_step_wise"] = V_weight_history_step_wise
        perf_counters["V_weight_history"] = V_weight_history
        perf_counters["SR_history"] = SR_history
        perf_counters["M_history"] = M_history
        perf_counters["M_history_step_wise"] = M_history_step_wise
        
    if enable_performance_counters: 
        pickle_file_name = f"{IMAGE_PATH}/perf_counters-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.bin"
        print("Saving performance counters to: ", pickle_file_name)
        with open(pickle_file_name, "wb") as f:
            # use pickle to save perf_counters
            pickle.dump(perf_counters, f)
    return M, V_weights, episode_rewards
   
#%%
# Part 1
#%%
original_goal=(1,1)
M, V, earned_rewards = actor_critic(np.eye(maze.size), n_steps=300, 
                                        alpha=0.05, gamma=0.99, n_episodes=1000, 
                                        goal=original_goal, start_func=normal_start)
part_1_one_hot_earned_rewards = earned_rewards
#%%
# plot state-value function
plt.figure(figsize=(10, 5))
plt.title(f'V(s): Goal {original_goal} Start {start}')
plot_maze(maze)
plt.imshow(V.reshape(maze.shape), cmap='hot')
plt.colorbar()
plt.savefig(f"{IMAGE_PATH}/values-part-1.png")
plt.show()

plt.figure(figsize=(10,5))
plt.title('Earned Discounted Rewards 1-Hot State Representation')
plt.plot(earned_rewards)
plt.plot(gaussian_filter(earned_rewards, 50), label='Rewards Smoothed')
plt.legend()
plt.savefig(f"{IMAGE_PATH}/earned_rewards-part-1.png")
plt.show()

#%% - Part 2, Now the same for an SR representation
original_goal=(1,1)
analytical_sr = random_walk_sr(transitions, 0.8).T
M, V, earned_rewards = actor_critic(analytical_sr, 
                                            n_steps=300, alpha=0.05, 
                                            gamma=0.99, n_episodes=1000,
                                            goal=original_goal, 
                                            start_func=normal_start)

part_2_sr_random_policy_earned_rewards = earned_rewards
#%%
# plot state-value function
plt.figure(figsize=(10, 5))
plot_maze(maze)
plt.title(f"V(s): goal at {goal} start at {start}")
plt.imshow(V.reshape(maze.shape), cmap='hot')
plt.colorbar()
plt.savefig(f"{IMAGE_PATH}/values-part-2.png")
plt.show()

#%%
plt.figure(figsize=(10, 5))
plt.plot(earned_rewards)
plt.plot(gaussian_filter(earned_rewards, 50), label='SR')
plt.plot(gaussian_filter(part_1_one_hot_earned_rewards, 50), label='1-Hot')
#plt.plot(gaussian_filter(np.square(part_1_one_hot_earned_rewards - earned_rewards), 2), label='Delta')
plt.legend()
plt.savefig(f"{IMAGE_PATH}/earned_rewards-part-2.png")
plt.show()
   
#%% Part-3
np.random.seed(42)

@numba.jit
def pick_random_element(arr):
    idx = np.random.randint(0, len(arr))
    return arr[idx]

def random_start(maze, goal):
    free_states = np.array([position_idx(i, j, maze) 
                        for i in range(maze.shape[0]) 
                            for j in range(maze.shape[1]) 
                                if check_legal(maze, (i, j)) and (i, j) != goal])
    return lambda: pick_random_element(free_states)

start_func = random_start(maze, goal)
learning_sr = random_walk_sr(transitions, 0.8).T
n_steps = 300 # 300 steps per episode
n_episodes = 1000  # explosion  in v_weights

alpha = 0.05 
gamma = 0.99 
sr_regularization=0.85
M, V, earned_rewards = actor_critic(learning_sr, n_steps, 
                                        alpha, gamma, n_episodes,
                                        update_sr=True, 
                                        start_func=start_func, 
                                        sr_regularization=sr_regularization, 
                                        debug=False,
                                        enable_performance_counters=True
                                        )
part_3_random_start_sr = earned_rewards

#%%
plt.figure(figsize=(10, 5))
plot_maze(maze)
plt.title(f"State-value function : goal at {goal}, max: {np.max(V):.2f}, min: {np.min(V):.2f}")
plt.imshow(V.reshape(maze.shape), cmap='hot', vmin=np.min(V), vmax=np.max(V))
plt.colorbar()
plt.savefig(f"{IMAGE_PATH}/values-part-3.png")
plt.show()
#%%
plt.figure(figsize=(10, 5))
plt.plot(earned_rewards)
plt.plot(gaussian_filter(earned_rewards, 30), label='random-start-sr')
plt.plot(gaussian_filter(part_2_sr_random_policy_earned_rewards, 30),label='fixed-start-sr')
plt.plot(gaussian_filter(part_1_one_hot_earned_rewards, 30), label='fixed-start-1-hot')
plt.legend()
plt.savefig(f"{IMAGE_PATH}/earned_rewards-part-3.png")
plt.show()

#%%
#%%
# Plot the SR of some states after this learning, also anything else you want.
#%% Plot the SR 
plt.plot(learning_sr[0, :])
plt.plot(learning_sr[1, :])

plt.figure(figsize=(20,15))
plt.imshow(learning_sr, cmap='hot')
plt.colorbar()
plt.savefig("sr-part-4.png")


original_random_walk_sr = np.copy(random_walk_sr(transitions, 0.8).T)
#%%
for state_idx in range(maze.size):
    if check_legal(maze, position_from_idx(state_idx, maze)):
        plt.figure(figsize=(10,5))
        delta = learning_sr[state_idx, :].reshape(maze.shape) - original_random_walk_sr[state_idx, :].reshape(maze.shape)
        total_error = np.sum(np.square(delta))
        plt.imshow(learning_sr[state_idx, :].reshape(maze.shape), cmap='hot')
        #plt.imshow(delta, cmap='hot')
        plt.title(f"SR for state {position_from_idx(state_idx,maze)} Random Walk Deviation:{total_error:.2f}")
        plt.colorbar()
        plt.savefig(f"{IMAGE_PATH}/sr-state-{state_idx}-part-4.png")
        plt.show()

#%% Part-4 Parallel
import numpy as np
from joblib import Parallel, delayed

# Define the function to be parallelized
def run_experiment(i, transitions, maze_shape, num_steps, num_episodes, original_goal, new_goal):
    print("Running experiment", i)
    # Run with updated SR
    re_learning_sr = random_walk_sr(transitions, 0.8).T
    start_func = random_start(maze, original_goal)

 
    # Run with random-walk SR
    analytical_sr = random_walk_sr(transitions, 0.8).T
    M, V, earned_rewards_clamped = actor_critic(
        analytical_sr, num_steps, 0.05, 0.99, num_episodes, 
        goal=new_goal,
        start_func=start_func
    )
    
    # Train to original goal
    _, _, _ = actor_critic(
        re_learning_sr, num_steps, 0.05, 0.99, num_episodes, 
        update_sr=True, goal=original_goal,
        start_func=start_func,
        sr_regularization=0.85,
    )
    
    # Learn new goal
    M, V, earned_rewards_relearned = actor_critic(
        re_learning_sr, num_steps, 0.05, 0.99, num_episodes, 
        update_sr=True, 
        goal=new_goal,
        start_func=start_func,
        sr_regularization=0.85
    )
    return earned_rewards_clamped, earned_rewards_relearned


# Parameters
goal = (5, 5)
original_goal = (1, 1)
new_goal = (5, 5)
goal_state = goal[0] * maze.shape[1] + goal[1]
num_episodes = 1000
num_steps = 400
earned_rewards_clamped_list = np.zeros((20, 400, 1000))
earned_rewards_relearned_list = np.zeros((20, 400, 1000))

# Parallel execution using Joblib
results = Parallel(n_jobs=-1)(delayed(run_experiment)(
    i, transitions, maze.shape, num_steps, num_episodes, original_goal, new_goal
) for i in range(20))

# Collect results
for i, (earned_rewards_clamped, earned_rewards_relearned) in enumerate(results):
    earned_rewards_clamped_list[i] = earned_rewards_clamped
    earned_rewards_relearned_list[i] = earned_rewards_relearned


#%% Save results with pickle
## Use timestamp and save pickl file for part-4
with open(f"{IMAGE_PATH}/part-4-results-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.bin", "wb") as f:
    pickle.dump((earned_rewards_clamped_list, earned_rewards_relearned_list), f)

#%%
# 20 - number of trials, 1000 number of episodes, 400 - number of steps
# Plot the performance averages of the two types of learners
avg_clamped = earned_rewards_clamped_list.mean(axis=1).mean(axis=0)
avg_relearned = earned_rewards_relearned_list.mean(axis=1).mean(axis=0)

plt.figure(figsize=(10, 5))
plt.plot(avg_clamped, label='clamped')
plt.plot(avg_relearned, label='relearned')
plt.legend()
plt.savefig(f"{IMAGE_PATH}/earned_rewards-part-4.png")
plt.show()
#%%
# Part 5
"""
Lastly, we will study how value initialization can aid in the learning of a
policy. 

The reward location is back at (1, 1), we always start at the 
original starting position, and use the SR from a random-walk policy as our
representation. 

So far, we have initialized our weights w with 0. 

Experiment with different initializations:
    along with both the 
        - 1-hot representation
        - SR. 

Try a couple of representative points (like 4-5 different values)
from 0 to 90 as your initialization. 

What do you observe, why do you think some values help 
while others hurt?

"""
# Reset Goal
goal = (1, 1)
goal_state = position_idx(goal[0], goal[1], maze)

#%%
# Run some learners with different value weight w initializations.
def gaussian_value_initialization(maze, goal, goal_value, std):
    """
    Maze represents the map of the environment. Goal represents the
    rewarding goal state, while goal value is the value of the goal state. 
    
    We set the mean of the 2-d gaussian distribution to the goal value, and 
    the standard deviation to std.
    """
    cols, rows = maze.shape
    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(rows), np.arange(cols))
    
    # Compute the Gaussian distribution centered at the goal
    goal_x, goal_y = goal
    value_weights = goal_value * np.exp(-((x - goal_x)**2 + (y - goal_y)**2) / (2 * std**2))
    print(value_weights.shape)
    retval = np.zeros(maze.size)
    for state_idx in range(maze.size):
        retval[state_idx] = value_weights[position_from_idx(state_idx, maze)[0], 
                                          position_from_idx(state_idx, maze)[1]]
    return retval 

plot_maze(maze)
plt.imshow(gaussian_value_initialization(maze, goal, goal_value, 5).reshape(maze.shape), cmap='hot') 
 
#%%
initialization_types = {
    "gaussian": lambda std: gaussian_value_initialization(maze, goal, goal_value, std),
    "constant": lambda c: np.ones(maze.size)*c,
}

initialization_type_args = {
    "gaussian": [(1,), (5,), (10,)],
    "constant": [(0,), (1,), (5,), (10,), (20,), (50,), (90,)],
}


one_hot_earned_rewards_map = {}
sr_earned_rewards_map = {}

for label, init_func in initialization_types.items():
    for args in initialization_type_args[label]:
        print(f"Running {label} with args {args}")
        v_init = init_func(*args)
        one_hot_earned_rewards = np.zeros((12,400))
        sr_earned_rewards = np.zeros((12,400))
        for i in range(12):
            M, V, earned_rewards = actor_critic(np.eye(maze.size), 300, 0.05, 0.99, 400, v_init=v_init, goal=goal)
            print("shape:",earned_rewards.shape)
            one_hot_earned_rewards[i] = earned_rewards
            analytical_sr = random_walk_sr(transitions, 0.8).T
            M, V, earned_rewards = actor_critic(analytical_sr, 300, 0.05, 0.99, 400, v_init=v_init, update_sr=False, goal=goal)
            sr_earned_rewards[i] = earned_rewards
        one_hot_earned_rewards_map[(label, args)] = one_hot_earned_rewards
        sr_earned_rewards_map[(label, args)] = sr_earned_rewards 
# plot the resulting learning curves
# %%
legend_templates = {
    "gaussian": "Gaussian std={}",
    "constant": "Constant c={}",
}

plt.figure(figsize=(10, 5))
filter_size = 7

for args in initialization_type_args["gaussian"]:
    plt.plot(gaussian_filter(one_hot_earned_rewards_map[("gaussian", args)].mean(axis=0), filter_size), label="1-Hot "+legend_templates["gaussian"].format(*args))
    plt.plot(gaussian_filter(sr_earned_rewards_map[("gaussian", args)].mean(axis=0), filter_size), label="SR " + legend_templates["gaussian"].format(*args), linestyle='--')
plt.title("Gaussian Value Initialization")
plt.legend()
plt.savefig(f"{IMAGE_PATH}/part-5-gaussian-value_initialization.png")
plt.show()

plt.figure(figsize=(10, 5))
for args in [(0,), (1,),  (10,), (50,), (90,)]: #initialization_type_args["constant"]:
    plt.plot(gaussian_filter(one_hot_earned_rewards_map[("constant", args)].mean(axis=0), filter_size), label="1-Hot "+legend_templates["constant"].format(*args))
    plt.plot(gaussian_filter(sr_earned_rewards_map[("constant", args)].mean(axis=0), filter_size), label="SR "+legend_templates["constant"].format(*args), linestyle='--')

plt.title("Constant Value Initialization")
plt.legend()
plt.savefig(f"{IMAGE_PATH}/part-5-constant-value_initialization.png")
plt.show()
# %%
