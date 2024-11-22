#%%
import numpy as np
import matplotlib.pyplot as plt
import logging
#%%
# define maze
# maze \in R^{9x13} with 
# 0s for free space and 1s for walls
maze = np.zeros((9, 13))

# place walls
maze[2, 6:10] = 1
maze[-3, 6:10] = 1
maze[2:-3, 6] = 1

# define start
start = (5, 7)

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

plot_maze(maze)
plt.scatter(start[1], start[0], marker='*', color='blue', s=100)
plt.tight_layout()
plt.savefig('maze.png')
plt.show()
#%%

####################################
############## Part 1 ##############
####################################
def is_inside_maze(maze, move):
    return move[0] >= 0 and move[0] < maze.shape[0] and move[1] >= 0 and move[1] < maze.shape[1]

def is_free_cell(maze, move):
    return maze[move[0], move[1]] == 0

def check_legal(maze, move):
    return is_inside_maze(maze, move) and is_free_cell(maze, move)

def reachable_moves(maze, pos):
    # return a list of all reachable moves from a given position
    legal_transitions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  
    possible_moves = [(pos[0] + move[0], pos[1] + move[1]) for move in legal_transitions]
    legal_moves = [move for move in possible_moves if check_legal(maze, move)]
    return legal_moves 

def select_random(moves, default_move):
    # return a random element from the list of moves
    return moves[np.random.randint(0, len(moves))] if len(moves) > 0 else  default_move

def random_walk(maze, start, n_steps):
    # Perform a single random walk in the 
    # given maze, starting from start, 
    # performing n_steps random moves
    #
    # moves into the wall and out of the 
    # maze boundary are not possible
    #print(f"Starting random walk at {start} for {n_steps} steps")
    # Initialize list to store positions
    positions = []
    pos = start
    for _ in range(n_steps+1):
        positions.append(pos)
        filtered_moves = reachable_moves(maze, pos)
        pos = select_random(filtered_moves, pos)
    # return a list of length n_steps + 1, containing the 
    # starting position and all subsequent locations as e.g. 
    # tuples or size (2) arrays 
    assert len(positions) == n_steps + 1
    return positions

def plot_path(maze, path):
    # Plot a maze and a path in it
    plot_maze(maze)
    path = np.array(path)
    plt.plot(path[:, 1], path[:, 0], c='red', lw=3)
    plt.scatter(path[0, 1], path[0, 0], marker='*', color='blue', s=100)
    plt.scatter(path[-1, 1], path[-1, 0], marker='*', color='green', s=100)
    plt.show()

# plot a random path
path = random_walk(maze, start, 40)
plot_path(maze, path)

#%%
####################################
############## Part 2 ##############
####################################
def learn_from_traj(succ_repr, trajectory, gamma=0.98, alpha=0.02):
    # Write a function to update a given successor representation 
    # (for the state at which the trajectory starts) 
    # using an example trajectory using discount factor 
    # gamma and learning rate alpha
    # return the updated successor representation
    start_state = trajectory[0]
    for next_state in trajectory[1:]: # Skip the first state
        for i in range(succ_repr.shape[0]):
            for j in range(succ_repr.shape[1]):
                occupancy_increment = 1 if (i, j) == next_state else 0
                succ_repr[i, j] = succ_repr[i, j] + \
                    alpha * (occupancy_increment + (gamma * succ_repr[next_state[0], next_state[1]]) - succ_repr[i, j])
    return succ_repr

# Initialize successor representation
succ_repr = np.zeros_like(maze)

# sample a whole bunch of trajectories 
# (reduce this number if this code takes too long, 
# but it shouldn't take longer than a minute with reasonable code)
for i in range(5001):
    # Sample a path (we use 340 steps here to sample states until the 
    # discounting becomes very small)
    path = random_walk(maze, start, 340) #AN: This is a full trajectory
    # update the successor representation
    succ_repr = learn_from_traj(succ_repr, path, alpha=0.02)  # choose a small learning rate

    # occasionally plot it
    if i in [0, 10, 100, 1000, 5000]:
        plot_maze(maze)
        plt.imshow(succ_repr, cmap='hot')
        if i == 5000:
             plt.savefig(f"empirical-{i}")
        plt.show()

#%%
####################################
############## Part 3 ##############
####################################
def position_idx(i, j, maze):
    return i * maze.shape[1] + j

def compute_transition_matrix(maze):
    # For a given maze, compute the transition 
    # matrix from any state to any other state under 
    # a random walk policy.  (You will need to 
    # think of a good way to map any 2D grid 
    # coordinates onto a single number for this).

    # Create a matrix over all state-pairs.
    num_states = maze.size 
    transitions = np.zeros((num_states, num_states)) 
    state_transition_counts = np.zeros(num_states)

    for i in range(maze.shape[0]): 
        for j in range(maze.shape[1]): 
            s = (i, j)
            s_idx = position_idx(i, j, maze) 
            for s_next in reachable_moves(maze, s): 
                s_n_idx = position_idx(s_next[0], s_next[1], maze)
                state_transition_counts[s_idx] += 1
                transitions[s_idx, s_n_idx] += 1
    # Normalize transitions
    transitions = transitions / state_transition_counts[:, None]
    transitions = np.nan_to_num(transitions)
    
    assert np.allclose(transitions.sum(axis=1), 1), "Rows of the transition matrix must sum to 1."
    assert np.all(transitions >= 0), "Transition matrix cannot have negative probabilities."

    return transitions 

#%%
def compute_transition_matrix_empirical(maze):
    # For a given maze, compute the transition 
    # matrix from any state to any other state under 
    # a random walk policy.  (You will need to 
    # think of a good way to map any 2D grid 
    # coordinates onto a single number for this).

    # Create a matrix over all state-pairs.
    #
    num_states = maze.size * maze.size  
    transitions = np.zeros((num_states, num_states)) 
    state_visit_counts = np.zeros(num_states)
    num_iterations = 10000

    for _ in range(num_iterations): 
        # Iterate over all states, filling in the 
        # transition probabilities to all other states 
        # on the next step (only one step into the future)
        trajectory = random_walk(maze, start, 340)  # 340 steps

        for s, s_next in zip(trajectory[:-1], trajectory[1:]): 
            i,j = s
            s_idx = position_idx(i, j, maze) 
            s_n_idx = position_idx(s_next[0], s_next[1], maze)
            state_visit_counts[s_idx] += 1
            transitions[s_idx, s_n_idx] += 1

    # Normalize transitions if neccessary.
    transitions = transitions / state_visit_counts[:, None]
    
    # Remove NaNs if necessary
    transitions = np.nan_to_num(transitions)
    # We will run the trajectories for n number of times
    # keep track of node visit counts as well as transition counts. 
    # normalize and return the probabilities as a matrix. 
    return transitions

#%%
####################################
############## Part 4 ##############
####################################
def compute_sr(transitions, i, j, gamma=0.98, shape=(9, 13)):
    # Given a transition matrix and a specific state 
    # (i, j), 
    # compute the successor representation of that state with 
    # discount factor gamma

    # initialize things (better to represent the current 
    # discounted occupancy as a vector here)
    num_states = shape[0] * shape[1]
    current_discounted_occupancy = np.zeros(num_states)
    current_discounted_occupancy[position_idx(i, j, maze)] = 1
    
    total = current_discounted_occupancy.copy()
    #TODO...

    # iterate for a number of steps
    convergence_error = 1e-6
    previous_total = total.copy()
    for i in range(340):
        total += gamma * (transitions @ total)
        convergence_error = np.linalg.norm(total - previous_total) 
        if i % 50 == 0: 
            print(f"Convergence Error: {convergence_error}")
        previous_total = total.copy()
    # return the successor representation, 
    # maybe reshape your 
    # vector into the maze shape now.
    return total.reshape(shape)

#%%
empirical_transitions = compute_transition_matrix_empirical(maze)
plt.imshow(empirical_transitions)
plt.show()
#%%
transitions = compute_transition_matrix(maze) 
#%%
plt.imshow(transitions, cmap='hot')
plt.savefig("transition_matrix.png")
plt.show()
#compute_transition_matrix(maze)
#%%
# compute state representation for start state
i, j = start
sr = compute_sr(transitions, i, j, 0.98, shape=maze.shape)

# plot state representation
plot_maze(maze)
plt.imshow(sr, cmap='hot')
plt.savefig("transition_iterate")
plt.show()


############################################
############## Part 5 (Bonus) ##############
############################################
# You're on your own now.
#%%
def compute_sr_bonus(transitions, i, j, gamma=0.98, shape=(9, 13)):
    # Given a transition matrix and a specific state 
    # (i, j), 
    # compute the successor representation of that state with 
    # discount factor gamma

    # initialize things (better to represent the current 
    # discounted occupancy as a vector here)
    num_states = shape[0] * shape[1]
    current_discounted_occupancy = np.zeros(num_states)
    current_discounted_occupancy[position_idx(i, j, maze)] = 1
    
    total = current_discounted_occupancy.copy()
    I = np.eye(num_states)
    transitions = np.linalg.inv(I - gamma * transitions) # LU ?
    total = transitions @ total
    return total.reshape(shape)
   
i, j = start
print(f"Computing SR Bonus: {i}, {j}")
sr_bonus = compute_sr_bonus(transitions, i, j, 
                            0.98, shape=maze.shape) 
plot_maze(maze)
plt.imshow(sr_bonus, cmap='hot')
# plt.savefig("transition_iterate")
plt.savefig("sr_bonus_start.png")
plt.show()

#%%
i, j = (start[0]-2, start[1])
print(f"Computing SR Bonus: {i}, {j}")
sr_bonus = compute_sr_bonus(transitions, i, j, 
                            0.98, shape=maze.shape) 
plot_maze(maze)
plt.imshow(sr_bonus, cmap='hot')
plt.savefig("sr_bonus_opposite_wall.png")
plt.show()
#%%
