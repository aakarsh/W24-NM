#%%
import numba
import numpy as np
import matplotlib.pyplot as plt

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
#%% :- TODO
@numba.jit
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
import numba

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
def init_propensities(maze, epsilon = 1e-5):
    M =  np.ones((maze.size, 4))* (-np.inf)
    
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            next_moves = possible_moves(maze, (i, j)) 
            reachable_moves = [(move, action) for action, move in enumerate(next_moves) if check_legal(maze, move)]
            for _, action in reachable_moves: 
                M[position_idx(i, j, maze), action] = epsilon
    return M

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
def actor_critic(state_representation, n_steps, 
                 alpha, 
                 gamma, 
                 n_episodes, 
                 update_sr=False, 
                 start_func=normal_start, 
                 v_init=0,
                 goal_reach_reward=goal_value, 
                 step_penalty=0):
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
    perf_counters = {"num_episodes": 0, "num_steps": 0, "num_goal_reached": 0}
    M = init_propensities(maze)
    # Initialize state-value function
    num_states = state_representation.shape[0]
    # w - weights for the value function
    V_weights = np.zeros(num_states) # TODO
    #
    earned_rewards = [] # TODO

    LEGAL_MOVES = legal_moves()

    # Iterate over episodes
    for _ in range(n_episodes):
        # Initializations
        # TODO
        # Move to the start state/possibly random start state
        perf_counters["num_episodes"] += 1
        state_idx = start_func()
        # cumulative discount factor
        I = 1
        # episode trajectory
        trajectory = []

        # Go until goal is reached
        for _ in range(n_steps):
            perf_counters["num_steps"] += 1
            # Act and Learn (Update both M and V_weights)
            
            # Compute action probabilities
            action_probabilities = softmax(M[state_idx, :])
            # Choose action according to action probabilities
            chosen_action = np.random.choice(4, p=action_probabilities)
            
            # take action
            move = LEGAL_MOVES[chosen_action]
            new_state = tuple(np.array(position_from_idx(state_idx, maze)) + np.array(move))
            i, j = new_state 
            
            if check_legal(maze, (i, j)): 
                trajectory.append(state_idx)
                goal_reached = (i, j) == goal
                if goal_reached:
                    perf_counters["num_goal_reached"] += 1

                V_state = V_weights @ state_representation[state_idx]
                # Compute the value of the new state, goal-state has value 0 
                new_state_idx = position_idx(i, j, maze)
                # V(s) = X(s) \cdot w || V(s) = 0 if s is goal
                V_new_state = V_weights @ state_representation[new_state_idx]  if not goal_reached else 0
                #  
                V_diff = ( gamma * V_new_state ) - V_state 
                reward = goal_reach_reward if goal_reached else step_penalty
                # TD error 
                delta = reward + V_diff
                # linear function \nabla_{V_weights} X(s)* V_weights  = X(s)
                V_weights += alpha * delta * state_representation[state_idx] 
                # Assuming same \alpha^\theta == \alpha^\theat) :( ?
                # Reduce the probability of the not-chosen action 
                M[state_idx, :] += alpha * I * delta * (-action_probabilities) 
                M[state_idx, chosen_action] += alpha * I * delta * (1) # so we have net (1 - action_probabilities[chosen_action]) increase in probability
               
                # Absorbing state  
                if (i, j) == goal: 
                    earned_rewards.append(I * reward)
                    if update_sr: # Update the state representation
                        for idx, state_idx in enumerate(trajectory):
                            current_trajectory = trajectory[idx:]
                            state_representation[state_idx, :] = \
                                learn_from_traj(state_representation[state_idx], 
                                                current_trajectory, gamma, alpha)
                    break # END EPISODE
                
                state_idx = new_state_idx 
                I *= gamma
            else: # no transition reward
                episode_reward += step_penalty
    print(perf_counters)
    return M, V_weights, earned_rewards


#%%
# One part to the solution of exercise part 3, if you want to update the 
# SR after each episode
def learn_from_traj(succ_repr, trajectory, gamma=0.98, alpha=0.05):
    # Write a function to update a given successor representation 
    # (for the state at which the trajectory starts) using an 
    # example trajectory using:
    #       discount factor gamma 
    #       learning rate alpha
    observed = np.zeros_like(succ_repr)
    for i, state in enumerate(trajectory):
        observed[state] += gamma ** i
    succ_repr += alpha * (observed - succ_repr)
    # Return the updated successor representation
    return succ_repr


# Part 1

#%%
M, V, earned_rewards = actor_critic(np.eye(maze.size), n_steps=300, 
                                        alpha=0.05, gamma=0.99, n_episodes=1000)

#%%
# plot state-value function
plot_maze(maze)
plt.imshow(V.reshape(maze.shape), cmap='hot')
plt.show()

plt.plot(earned_rewards)
plt.show()


# Part 2, Now the same for an SR representation
#%%
M, V, earned_rewards = actor_critic(analytical_sr_read_only, n_steps=300, alpha=0.05, gamma=0.99, n_episodes=1000)

#%%
# plot state-value function
plot_maze(maze)
plt.imshow(V.reshape(maze.shape), cmap='hot')
plt.show()

plt.plot(earned_rewards)
plt.show()

#%%
# Part 3
def random_start(maze):
    def pick_start():
        # Suggested encoding of 2-D location onto states.
        i, j = (1, 1)
        while True:
            i, j = np.random.randint(maze.shape[0]), np.random.randint(maze.shape[1])
            if check_legal(maze, (i, j)):
                break
        state = position_idx(i,j, maze) 
        return state
    # Define yourself a function to return a random (non-wall) starting state 
    # to pass into the actor_critic function.
    return pick_start

#%%
plt.hist([random_start(maze)() for i in range(100000)], bins=100)

#%%
start_func = random_start(maze)
learning_sr = random_walk_sr(transitions, 0.8).T
n_steps = 1000 # 300 steps per episode
n_episodes = 5000 # 1000 episodes
M, V, earned_rewards = actor_critic(learning_sr, n_steps, 0.05, 0.99, n_episodes,
                                       update_sr=True, start_func=start_func)

plot_maze(maze)
plt.imshow(V.reshape(maze.shape), cmap='hot')
plt.show()

#%%
plt.plot(earned_rewards)
plt.show()

#%%
# Plot the SR of some states after this learning, also anything else you want.
# TODO:-
# Part 4
#%% Plot the SR 
plt.plot(learning_sr[0, :])
plt.plot(learning_sr[1, :])
plt.imshow(learning_sr, cmap='hot')

#%%
for state_idx in range(maze.size):
    plt.figure()
    plt.imshow(learning_sr[state_idx, :].reshape(maze.shape), cmap='hot')
    plt.title(f"SR for state {position_from_idx(state_idx,maze)}, goal at {goal}")
    plt.colorbar()
    plt.show()
#%% Plot the SR 

TODO
goal = (5, 5)
goal_state = goal[0]*maze.shape[1] + goal[1]
for i in range(20):

    # run with random walk SR
    M, V, earned_rewards_clamped = actor_critic(TODO, 300, 0.05, 0.99, 400)
    TODO

    # run with updated SR
    M, V, earned_rewards_relearned = actor_critic(TODO, 300, 0.05, 0.99, 400)
    TODO

# plot the performance averages of the two types of learners
TODO


# Part 5

# reset goal
goal = (1, 1)
goal_state = goal[0]*maze.shape[1] + goal[1]

# run some learners with different value weight w initializations

TODO
for v_inits in [TODO]:
    TODO
    for i in range(12):

        M, V, earned_rewards = actor_critic(TODO, 300, 0.05, 0.99, 400)
        TODO
        M, V, earned_rewards = actor_critic(TODO, 300, 0.05, 0.99, 400)
        TODO

# plot the resulting learning curves
# %%
