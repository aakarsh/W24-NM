#%%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.ndimage import gaussian_filter
import datetime
import matplotlib.pyplot as plt
import numba
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns 

import sys
sys.path.append('/Users/aakarsh/src/TUE-WINTER-2024/dropped/W24-NEURO-MODELING/hws/hw-5/')
import importlib
import maze_utils
importlib.reload(maze_utils)
from maze_utils import make_maze, plot_maze, plot_trajectory

#%%
dump_file_path = '/Users/aakarsh/src/TUE-WINTER-2024/dropped/W24-NEURO-MODELING/hws/hw-5/sr-ac/images/'
debug_file=f'{dump_file_path}/perf_counters-2024-12-26-22-30-11.bin'
#%%
# find most reacent perf_counters file
def find_most_recent_file(dump_file_path):
    files = os.listdir(dump_file_path)
    files = [f for f in files if 'perf_counters' in f]
    files = sorted(files, reverse=True)
    return f"{dump_file_path}/{files[0]}"

# %% 
# Parse pickle file
def parse_pickle_file(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

counters = parse_pickle_file(find_most_recent_file(dump_file_path)) 
#%%
counters['episode_counters'][24]['trajectory']
#%%
maze_vars = make_maze()
maze = maze_vars['maze']
start = maze_vars['start']

goal = maze_vars['goal']
goal_state = maze_vars['goal_state']
goal_value = maze_vars['goal_value']

#%%
plot_maze(maze)
#plot_trajectory(maze, counters['episode_counters'][24]['trajectory'])
#%%
# plot trajectory
# TODO: Check sr before and after trajectory processing.
#%%

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_maze_values(v_weights_history, sr_history, maze_shape, output_path='maze_values_evolution.mp4'):
    num_episodes, num_states = v_weights_history.shape
    _, _, sr_states = sr_history.shape

    if num_states != sr_states:
        raise ValueError("Mismatch between number of states in V_weights and SR_history.")

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(maze_shape[1] * 2, maze_shape[0] * 2))  # Bigger figure
    maze_values = sr_history[0] @ v_weights_history[0]
    maze_values_grid = maze_values.reshape(maze_shape)
    im = ax.imshow(maze_values_grid, cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(maze_values_grid))
    ax.set_title("Evolution of Maze Cell Values", fontsize=16)
    ax.set_xlabel("Maze Columns", fontsize=12)
    ax.set_ylabel("Maze Rows", fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Value", fontsize=12)

    # Add text annotations to each cell
    text_annotations = []
    for i in range(maze_shape[0]):
        for j in range(maze_shape[1]):
            text = ax.text(j, i, f"{maze_values_grid[i, j]:.2f}",
                           ha='center', va='center', color='white', fontsize=10)
            text_annotations.append(text)

    def update(frame):
        # Compute the grid values for the current episode
        maze_values = sr_history[frame] @ v_weights_history[frame]
        maze_values_grid = maze_values.reshape(maze_shape)
        im.set_array(maze_values_grid)
        ax.set_title(f"Episode {frame}", fontsize=16)

        # Update text annotations
        for i in range(maze_shape[0]):
            for j in range(maze_shape[1]):
                idx = i * maze_shape[1] + j
                text_annotations[idx].set_text(f"{maze_values_grid[i, j]:.2f}")

        return [im] + text_annotations

    ani = animation.FuncAnimation(fig, update, frames=range(num_episodes), interval=200, blit=True)
    ani.save(output_path, fps=10, writer='ffmpeg')
    plt.close(fig)

animate_maze_values(counters['V_weight_history'], counters['SR_history'], maze.shape, output_path='maze_values_evolution.mp4')
#%%

def animate_sr_tiles(sr_history, maze_shape, output_path='sr_evolution.mp4'):
    num_episodes, num_states, _ = sr_history.shape

    if num_states != np.prod(maze_shape):
        raise ValueError("Mismatch between number of states and maze size.")

    # Create figure and axes
    fig, axes = plt.subplots(maze_shape[0], maze_shape[1], figsize=(maze_shape[1] * 2, maze_shape[0] * 2))
    axes = axes.flatten()
    ims = []

    # Initialize heatmaps for each state
    for ax in axes:
        im = ax.imshow(np.zeros((maze_shape[0], maze_shape[1])), cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        ims.append(im)
        ax.axis('off')  # Turn off axes for clean display

    fig.suptitle("Evolution of Successor Representations", fontsize=16)

    def update(frame):
        for state_idx, ax in enumerate(axes):
            sr_values = sr_history[frame, state_idx, :].reshape(maze_shape)
            ims[state_idx].set_array(sr_values)
        fig.suptitle(f"Episode {frame}", fontsize=16)
        return ims

    ani = animation.FuncAnimation(fig, update, frames=range(num_episodes), interval=200, blit=True)
    ani.save(output_path, fps=10, writer='ffmpeg')
    plt.close(fig)

print("run")
counters = parse_pickle_file(find_most_recent_file(dump_file_path))
animate_sr_tiles(counters['SR_history'], maze.shape, output_path='sr_evolution.mp4')
#%%
