import numpy as np
import sys

sys.path.append('/Users/aakarsh/src/TUE-WINTER-2024/dropped/W24-NEURO-MODELING/hws/hw-5/')
import maze_utils

import importlib
importlib.reload(maze_utils)

from maze_utils import (make_maze, 
                        plot_maze, 
                        plot_trajectory, 
                        learn_from_traj, 
                        update_sr_after_episode)

def test_learn_from_traj_basic():
    """
    """
    # Basic test with a simple trajectory
    succ_repr = np.zeros(6)
    trajectory = [0, 1, 2, 3]
    gamma, alpha = 0.9, 0.1
    updated_repr = learn_from_traj(np.copy(succ_repr), trajectory, gamma, alpha)
    # Compute expected values manually
    expected = np.zeros(6)
    for i, state in enumerate(trajectory):
        expected[state] += gamma ** i
    expected = succ_repr + alpha * (expected - succ_repr)
    assert np.allclose(updated_repr, expected), "Basic test failed"

def test_update_sr_after_episode():
    """
    """
    # Define a simple SR matrix with a 3x3 maze (9 states)
    state_representation = np.zeros((9, 9))  # Initial SR as zeros
    trajectory = [0, 1, 2]  # Simple trajectory from state 0 -> 1 -> 2
    gamma = 0.9  # Discount Factor
    alpha = 0.1  # Learning Rate
    
    # Expected changes
    # For state 0, it should observe [1, gamma, gamma^2] along the trajectory
    expected_sr_0 = alpha * np.array([1, gamma, gamma**2, 0, 0, 0, 0, 0, 0])
    expected_sr_1 = alpha * np.array([0, 1, gamma, 0, 0, 0, 0, 0, 0])
    expected_sr_2 = alpha * np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    
    # Call the function
    updated_sr = update_sr_after_episode(state_representation, 
                                         trajectory, 
                                         gamma, 
                                         alpha, 
                                         debug=False)
        
    # Check if the updated SR matches the expected SR for state 0
    actual_sr_0 = updated_sr[0, :]
    actual_sr_1 = updated_sr[1, :]
    actual_sr_2 = updated_sr[2, :]
    
    assert np.allclose(actual_sr_1, expected_sr_1),  \
        "Test failed: SR update for state 1 does not match expected values"
    assert np.allclose(actual_sr_0, expected_sr_0), \
        "Test failed: SR update for state 0 does not match expected values"
    assert np.allclose(actual_sr_2, expected_sr_2), \
        "Test failed: SR update for state 0 does not match expected values"