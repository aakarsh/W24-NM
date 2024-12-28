#%%
from datetime import datetime
import os
import numba
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

def make_maze():
    """
    Create a maze environment
    """
    maze = np.zeros((9, 13))
    maze[2, 6:10] = 1
    maze[-3, 6:10] = 1
    maze[2:-3, 6] = 1
        
    start = (5, 7)
    goal = (1, 1)
    goal_state = goal[0]*maze.shape[1] + goal[1]
    goal_value = 10
    
    return {
        'maze': maze,
        'start': start,
        'goal': goal,
        'goal_state': goal_state,
        'goal_value': goal_value
    }
    
def plot_maze(maze):
    plt.imshow(maze, cmap='binary')

    # draw thin grid
    for i in range(maze.shape[0]):
        plt.plot([-0.5, maze.shape[1]-0.5], [i-0.5, i-0.5], c='gray', lw=0.5)
    for i in range(maze.shape[1]):
        plt.plot([i-0.5, i-0.5], [-0.5, maze.shape[0]-0.5], c='gray', lw=0.5)

    plt.xticks([])
    plt.yticks([])

def plot_trajectory(maze, trajectory):
    plot_maze(maze)
    trajectory = np.array(trajectory)
    # map trajctory indices to 2D coordinates
    trajectory = np.array([ [position_from_idx(pos, maze)] for pos in trajectory]) 
    print("trajectory: ", trajectory)
    plt.plot(trajectory[:, 1], trajectory[:, 0], c='red', lw=2)
    plt.plot(trajectory[0, 1], trajectory[0, 0], c='green', marker='o')
    plt.plot(trajectory[-1, 1], trajectory[-1, 0], c='red', marker='o')

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


def analytical_sr(transitions, gamma):
    return np.linalg.inv(np.eye(transitions.shape[0]) - gamma * transitions.T)

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

def plot_sr_history(SR_history, episode_idx, state_idx):
    """
    Plot the history of SR over time and states
    """
    print("Plotting SR history")
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

   
    # Define the axes
    time = np.arange(SR_history.shape[0])  # Episode indices
    states = np.arange(SR_history[episode_idx, state_idx, :].shape[0])  # State indices
    time, states = np.meshgrid(time, states)  # Create meshgrid for 3D plotting
    print(f"time.T.shape: {time.T.shape}")
    print(f"states.shape: {states.T.shape}")
    print(f"SR_history[episode_idx, state_idx, :].shape: {SR_history[episode_idx, state_idx, :].shape}")
    print(f"SR_history.shape: {SR_history.shape}")
    print(f"SR_history[episode_idx, state_idx, :].shape: {SR_history[episode_idx, state_idx, :].shape}")
    print(f"SR_history[episode_idx, state_idx, :].shape: {SR_history[episode_idx, state_idx, :].shape}")
     
    # Plot the surface of SR over time and states
    ax.set_title(f"Episode {episode_idx} State: {state_idx} {position_from_idx(state_idx, maze)}- SR History")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("States")
    ax.set_zlabel("SR")
    
    print(f"time.T.shape: {time.T.shape}")
    print(f"states.shape: {states.T.shape}")
    print(f"SR_history.shape: {SR_history.shape}")
    
    ax.plot_surface(time.T, states.T, SR_history[:, state_idx, :], cmap='viridis')
    
    # Save the figure
    plt.savefig(f"{IMAGE_PATH}/sr-history-episode-{episode_idx}-state-{state_idx}-{position_from_idx(state_idx, maze)}.png")
    plt.close(fig)  # Close the figure to free up memory

#%%
def plot_stepwise_v_weights_history(V_weight_history, episode_idx,step_idx):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    time = np.arange(V_weight_history[episode_idx].shape[0])  # steps indices
    states = np.arange(V_weight_history[episode_idx].shape[1])  # State indices
    time, states = np.meshgrid(time, states)  # Create meshgrid for 3D plotting    
  
    print(f"time.T.shape: {time.T.shape}")
    print(f"states.shape: {states.T.shape}")
    print(f"V_weight_history.shape: {V_weight_history[episode_idx].shape}")
    ax.set_title(f"Episode {episode_idx} - V-Weights History - {goal} - Step {step_idx}")
    ax.set_xlabel("Steps")
    ax.set_ylabel("States")
    ax.set_zlabel("V-Weights")
    
    ax.plot_surface(time.T, states.T, V_weight_history[episode_idx], cmap='viridis')
    plt.savefig(f"{IMAGE_PATH}/v-weights-per-step-history-episode-{episode_idx}-{step_idx}.png") 
    plt.close(fig)  # Close the figure to free up memory
    
    
#%%    
def plot_v_weights(V_weight_history, episode_idx):
    # Plot the history of V_weights over time and states
    #
    # Assuming V_weight_history is a 2D array (episodes x states)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    # Define the axes
    time = np.arange(V_weight_history.shape[0])  # Episode indices
    states = np.arange(V_weight_history.shape[1])  # State indices
    time, states = np.meshgrid(time, states)  # Create meshgrid for 3D plotting

    # Plot the surface of V_weights over time and states
    # print(f"V_weight_history.shape: {V_weight_history.T.shape}")
    print(f"time.T.shape: {time.T.shape}")
    print(f"states.shape: {states.T.shape}")
    print(f"V_weight_history.shape: {V_weight_history.shape}") 

    ax.set_title(f"Episode {episode_idx} - V-Weights History")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("States")
    ax.set_zlabel("V-Weights")

    ax.plot_surface(time.T, states.T, V_weight_history, 
                    cmap='viridis')
    # Save the figure
    plt.savefig(f"{IMAGE_PATH}/v-weights-history-episode-{episode_idx}.png")
    plt.close(fig)  # Close the figure to free up memory


#%%
# One part to the solution of exercise part 3, if you want to update the 
# SR after each episode
def learn_from_traj(succ_repr, trajectory, gamma=0.98, alpha=0.05, debug=False):
    # Write a function to update a given successor representation 
    # (for the state at which the trajectory starts) using an 
    # example trajectory using:
    #       discount factor gamma 
    #       learning rate alpha
    observed = np.zeros_like(succ_repr)
    #if debug: print(f"learn_from_traj: {trajectory}") 
    for i, state in enumerate(trajectory):
        #if debug: print(f"learn_from_traj: state: {state}")
        observed[state] += gamma ** i
    assert (observed >= 0).all(), "observed should be positive"
    #assert (succ_repr >= 0).all(), "succ_repr should be positive"
    delta =  observed - succ_repr
    #if debug: print(f"delta: {delta}")
    #if debug: print(f"successor: {succ_repr}, observed: {observed}, total-delta:{np.sum(np.square(delta))}")
    # if np.sum(np.abs(delta)) >0 : print("non-zero-delta: ", np.sum(np.abs(delta)))
    succ_repr += alpha * delta 
    # assert (succ_repr >= 0).all(), "succ_repr should remain positive"
    # Return the updated successor representation
    return succ_repr

def update_sr_after_episode(state_representation, trajectory, 
                            gamma=0.98, alpha=0.05, 
                            debug=False, 
                            regularization=0.05):
    old_state_representation = np.copy(state_representation)
    trajectory = np.array(trajectory)
    new_state_representation = np.copy(state_representation) 
    for idx, state_idx in enumerate(trajectory):
        if debug: 
            print(f"state_idx: {state_idx}") 
        current_trajectory = trajectory[idx:]
        if debug: 
            print(f"current_trajectory: {current_trajectory}")
        # Update the state representation for the state at which the trajectory starts
        updated_values = learn_from_traj(state_representation[state_idx, :], 
                            current_trajectory, gamma, alpha, 
                            debug=debug)
 
        new_state_representation[state_idx, :] = \
            (1 - regularization) * updated_values + \
            regularization * old_state_representation[state_idx, :]

       
    if debug: 
        # Figure with row of three images
        plt.figure(figsize=(15, 30))
        ax = plt.subplot(1, 3, 1)
        ax.set_title("Old State Representation")
        plt.imshow(old_state_representation, cmap='hot')
        plt.colorbar()
        
        ax = plt.subplot(1, 3, 2)
        ax.set_title("New State Representation")
        plt.imshow(state_representation, cmap='hot')
        plt.colorbar()
        
        ax = plt.subplot(1, 3, 3)
        ax.set_title(f"Change: Trajectory: {trajectory}")
        plt.imshow(state_representation - old_state_representation, cmap='hot')
        plt.colorbar()
        plt.show()

    if debug:
        for i in range(state_representation.shape[0]):
            for j in range(state_representation.shape[1]):
                if np.abs(state_representation[i, j] - old_state_representation[i, j]) > 0:
                    if debug: print(f"state: {i, j}, old: {old_state_representation[i, j]}, new: {state_representation[i, j]}")
                    if debug: print(f"difference: {state_representation[i, j] - old_state_representation[i, j]}")
                #if debug: print(f"Altered transition from {position_from_idx(i, maze)} to {position_from_idx(j, maze)}")
                
    #assert not np.allclose(old_state_representation, state_representation)
    total_change = np.sum(np.abs(old_state_representation - state_representation))
    if debug: 
        print(f"total-change: {total_change}")
    return new_state_representation
 