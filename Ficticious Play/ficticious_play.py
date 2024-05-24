import numpy as np
import matplotlib.pyplot as plt

# Constants for grid dimensions
GRID_WIDTH = 3
GRID_HEIGHT = 3

# Initial positions
P1_INIT_POS = [2, 1]
P2_INIT_POS = [0, 1]

def initialize_grid():
    """Initialize the game grid with players at their start positions."""
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    grid[P1_INIT_POS[0], P1_INIT_POS[1]] = 1  # Player 1 position
    grid[P2_INIT_POS[0], P2_INIT_POS[1]] = 2  # Player 2 position
    return grid

def move_player(current_pos, action):
    """Move player based on the action."""
    x, y = current_pos
    if action == "left" and y > 0:
        y -= 1
    elif action == "right" and y < GRID_WIDTH - 1:
        y += 1
    return [x, y]

def drop_object(p2_pos):
    """Determine the column where the object dropped by P2 will land."""
    y = p2_pos[1]
    rand_num = np.random.rand()

    if y == 0:  # Leftmost column
        if rand_num < 0.85:
            return y
        else:
            return y + 1
    elif y == 1:  # Middle column
        if rand_num < 0.15:
            return y - 1
        elif rand_num < 0.85:
            return y
        else:
            return y + 1
    else:  # Rightmost column
        if rand_num < 0.15:
            return y - 1
        else:
            return y

def compute_reward(p1_pos, drop_col):
    """Compute rewards for players based on P1's position and drop column."""
    if p1_pos[1] == drop_col:
        return +10, -10  # P1 catches, so +10 for P1 and -10 for P2
    else:
        return -10, +10  # P1 misses, so -10 for P1 and +10 for P2


# Constants for possible actions and grid dimensions
ACTIONS = ['left', 'right', 'stay']
GRID_WIDTH = 3
GRID_HEIGHT = 3

# Initial positions
P1_INIT_POS = [2, 1]
P2_INIT_POS = [0, 1]

# Initialize beliefs about the other player's strategies.
# Belief dimensions: GRID_WIDTH x len(ACTIONS)
belief_P1_about_P2 = np.ones((GRID_WIDTH, len(ACTIONS))) / len(ACTIONS)  # Uniform distribution initially
belief_P2_about_P1 = np.ones((GRID_WIDTH, len(ACTIONS))) / len(ACTIONS)

# Initialize Q tables for both players.
# Q table dimensions: GRID_WIDTH (P1 position) x GRID_WIDTH (P2 position) x len(ACTIONS)
Q_P1 = np.zeros((GRID_WIDTH, GRID_WIDTH, len(ACTIONS)))
Q_P2 = np.zeros((GRID_WIDTH, GRID_WIDTH, len(ACTIONS)))

# Choosing actions based on Equation 3.1
def choose_action(player, state, epsilon=0.1):
    """Action selection based on expected Q-value."""
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)

    if player == "P1":
        expected_Qs = np.dot(Q_P1[state[0], state[1], :], belief_P1_about_P2[state[1]])
        return ACTIONS[np.argmax(expected_Qs)]
    else:  # P2
        expected_Qs = np.dot(Q_P2[state[1], state[0], :], belief_P2_about_P1[state[0]])
        return ACTIONS[np.argmax(expected_Qs)]

# Updating belief using Equation 3.2
def update_belief(player, state, observed_action):
    alpha = 0.1  # Learning rate
    action_index = ACTIONS.index(observed_action)

    if player == "P1":
        belief_P1_about_P2[state[1], action_index] = (1 - alpha) * belief_P1_about_P2[state[1], action_index] + alpha
        # Normalize the beliefs so they sum to 1
        belief_P1_about_P2[state[1], :] /= np.sum(belief_P1_about_P2[state[1], :])
    else:  # P2
        belief_P2_about_P1[state[0], action_index] = (1 - alpha) * belief_P2_about_P1[state[0], action_index] + alpha
        # Normalize the beliefs so they sum to 1
        belief_P2_about_P1[state[0], :] /= np.sum(belief_P2_about_P1[state[0], :])

# Updating Q-values using Equation 3.3
def update_q_value(player, state, action, reward, new_state, gamma=0.95, beta=0.1):
    """Update the Q-value for a given state and action based on reward and new state."""
    action_index = ACTIONS.index(action)
    
    if player == "P1":
        max_q_value_next_state = np.max(Q_P1[new_state[0], new_state[1], :])
        Q_P1[state[0], state[1], action_index] += beta * (reward + gamma * max_q_value_next_state - Q_P1[state[0], state[1], action_index])
    else:  # P2
        max_q_value_next_state = np.max(Q_P2[new_state[1], new_state[0], :])
        Q_P2[state[1], state[0], action_index] += beta * (reward + gamma * max_q_value_next_state - Q_P2[state[1], state[0], action_index])


def main_game_loop(num_episodes=1000, gamma=0.95, beta=0.1, epsilon=0.1):
    #rewards_P1 = []
    #rewards_P2 = []
    # Lists to track cumulative rewards
    cumulative_rewards_P1 = []
    cumulative_rewards_P2 = []
    
    # Variables to hold the cumulative sum
    cum_reward_P1 = 0
    cum_reward_P2 = 0
    for episode in range(num_episodes):
        # Reset to initial positions
        p1_pos = P1_INIT_POS.copy()
        p2_pos = P2_INIT_POS.copy()

        # Both players choose actions
        p1_action = choose_action("P1", (p1_pos[1], p2_pos[1]), epsilon)
        p2_action = choose_action("P2", (p1_pos[1], p2_pos[1]), epsilon)
        
        # Move players based on chosen actions
        new_p1_pos = move_player(p1_pos, p1_action)
        new_p2_pos = move_player(p2_pos, p2_action)

        # Determine the drop column for P2's object and compute rewards
        drop_col = drop_object(new_p2_pos)
        p1_reward, p2_reward = compute_reward(new_p1_pos, drop_col)

        # Update Q-values for both players
        update_q_value("P1", (p1_pos[1], p2_pos[1]), p1_action, p1_reward, (new_p1_pos[1], new_p2_pos[1]), gamma, beta)
        update_q_value("P2", (p1_pos[1], p2_pos[1]), p2_action, p2_reward, (new_p1_pos[1], new_p2_pos[1]), gamma, beta)
        
        # Update beliefs based on observed actions
        update_belief("P1", (p1_pos[1], p2_pos[1]), p2_action)
        update_belief("P2", (p1_pos[1], p2_pos[1]), p1_action)

        # Record rewards for this episode
        #rewards_P1.append(p1_reward)
        #rewards_P2.append(p2_reward)

        cum_reward_P1 += p1_reward
        cum_reward_P2 += p2_reward

        cumulative_rewards_P1.append(cum_reward_P1)
        cumulative_rewards_P2.append(cum_reward_P2)

    return cumulative_rewards_P1, cumulative_rewards_P2

# Run the main game loop
cumulative_rewards_P1, cumulative_rewards_P2 = main_game_loop(num_episodes=15000)

# Display results
print("Q-values for Player 1:")
print(Q_P1)

print("\nQ-values for Player 2:")
print(Q_P2)

print("\nBelief of Player 1 about Player 2's strategy:")
print(belief_P1_about_P2)

print("\nBelief of Player 2 about Player 1's strategy:")
print(belief_P2_about_P1)

# Plot cumulative rewards
plt.figure(figsize=(12, 6))
plt.plot(cumulative_rewards_P1, label='Player 1 Cumulative Rewards', alpha=0.6)
plt.plot(cumulative_rewards_P2, label='Player 2 Cumulative Rewards', alpha=0.6)
plt.ylabel('Cumulative Reward')
plt.xlabel('Episode')
plt.title('Cumulative Rewards Over Episodes')
plt.legend()
plt.grid(True)
plt.show()

# Plot beliefs of Player 1 about Player 2's strategy
for idx, action in enumerate(ACTIONS):
    plt.plot(belief_P1_about_P2[:, idx], label=f"Belief about {action}")

plt.xlabel("Position")
plt.ylabel("Belief")
plt.title("Beliefs of Player 1 about Player 2's Strategy")
plt.legend()
plt.show()
