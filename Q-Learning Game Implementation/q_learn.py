### Libraries to be imported
import numpy as np
import pygame
import matplotlib.pyplot as plt

class GameVisualizer:
    def __init__(self, GRID_WIDTH, GRID_HEIGHT):
        self.GRID_WIDTH = GRID_WIDTH
        self.GRID_HEIGHT = GRID_HEIGHT
        
        pygame.init()
        self.CELL_SIZE = 40
        self.BALL_RADIUS = 15
        self.BALL_COLOR = (255, 0, 255)
        self.delay_time = 200
        self.WHITE = (255, 255, 255)

        # Load images for visualization
        self.basket_image = pygame.image.load("basket.png")
        self.basket_image = pygame.transform.scale(self.basket_image, (self.CELL_SIZE, self.CELL_SIZE))
        self.bg_img = pygame.image.load("background.png")
        self.bg_img = pygame.transform.scale(self.bg_img, (GRID_WIDTH * self.CELL_SIZE, GRID_HEIGHT * self.CELL_SIZE))

        self.screen = pygame.display.set_mode((GRID_WIDTH * self.CELL_SIZE, GRID_HEIGHT * self.CELL_SIZE))
        pygame.display.set_caption('Game of Catch')

    def draw_grid(self):
        for x in range(0, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.WHITE, (x, 0), (x, self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(0, self.GRID_HEIGHT * self.CELL_SIZE, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.WHITE, (0, y), (self.GRID_WIDTH * self.CELL_SIZE, y))
    
    def visualize(self, env, catches, losses, total_reward):
        # Clear the screen
        self.screen.blit(self.bg_img, (0, 0))
        self.draw_grid()
        # Draw ball and basket based on env state
        for i in range(self.GRID_HEIGHT):
            for j in range(self.GRID_WIDTH):
                if env[i, j] == 1:
                    pygame.draw.circle(self.screen, self.BALL_COLOR, (j*self.CELL_SIZE + self.CELL_SIZE//2, i*self.CELL_SIZE + self.CELL_SIZE//2), self.BALL_RADIUS)
                elif env[i, j] == 2:
                    self.screen.blit(self.basket_image, (j*self.CELL_SIZE, i*self.CELL_SIZE))
        
        font = pygame.font.Font(None, 14)
        result_text = f"Catches: {catches}\nLosses: {losses}\nTotal Reward: {total_reward}"
        text_lines = result_text.split('\n')
        for index, line in enumerate(text_lines):
            text = font.render(line, True, (0, 0, 0))  # Black color for text
            self.screen.blit(text, (self.GRID_WIDTH * self.CELL_SIZE - 200, 20 + index * 40)) 
        pygame.display.flip()  # Update the screen
        pygame.time.wait(self.delay_time)  # Introduce delay to control speed

class GameOfCatch:
    def __init__(self,GRID_WIDTH,GRID_HEIGHT, NUM_EPISODES):
        # Initialization
        self.NUM_EPISODES = NUM_EPISODES
        self.GRID_WIDTH = GRID_WIDTH
        self.GRID_HEIGHT = GRID_HEIGHT
        self.env = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        self.epsilon = 0.85

        # Q Learning
        self.q_table = np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_WIDTH, 3))
        self.alpha = 0.19
        self.gamma = 0.8
        self.errors = []

    def update_environment(self,env, ball_row, ball_col, basket_col, action, ball_speed=1):
        movement_penalty = -0.1

        # Update the ball's position based on ball_speed
        ball_row = min(self.GRID_HEIGHT - 1, ball_row + ball_speed)  # Ensure it doesn't go off-grid
    
        # Update basket's position based on action
        if action == 0:  # Move left
            basket_col = max(0, basket_col - 1)  # Ensure it doesn't go off-grid
            reward = movement_penalty
        elif action == 1:  # Move right
            basket_col = min(self.GRID_WIDTH - 1, basket_col + 1)  # Ensure it doesn't go off-grid
            reward = movement_penalty
        else:  # Stay
            reward = 0  # No penalty for staying

        # Check for catching or missing the ball
        if ball_row == self.GRID_HEIGHT - 1:  # Ball is at the last row
            if ball_col == basket_col:
                reward += 10  # Caught the ball
            else:
                reward -= 10  # Missed the ball

        # Update the environment grid
        env.fill(0)
        env[ball_row, ball_col] = 1  # Ball's position
        env[self.GRID_HEIGHT - 1, basket_col] = 2  # Basket's position (always at the bottom row)

        new_state = (ball_row, ball_col, basket_col)
        return new_state, reward

    def choose_action(self, state, epsilon):
        # If a randomly chosen value is less than epsilon, take a random action (explore)
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice([0, 1, 2])  # Three possible actions: left, right, stay
        else:
            # Take the action with the highest Q-value for the current state (exploit)
            ball_row, ball_col, basket_col = state
            action = np.argmax(self.q_table[ball_row, ball_col, basket_col, :])
        return action

    def q_learning(self, state, action, reward, new_state):
        # Retrieve current Q-value
        current_q = self.q_table[state[0], state[1], state[2], action]

        # Calculate the expected Q-value for the best action in the new state
        expected_q = np.max(self.q_table[new_state[0], new_state[1], new_state[2], :])

        # Calculate the Temporal Difference (TD)
        td = reward + self.gamma * expected_q - current_q

        # Update the Q-value using the Q-learning formula
        self.q_table[state[0], state[1], state[2], action] += self.alpha * td

        # Store the TD for visualization
        self.errors.append(td)

    def plot_errors(self,errors, window_size=100):
        # Compute the average TD error for each episode
        steps_per_episode = len(errors) // self.NUM_EPISODES
        episode_errors = [np.mean(errors[i:i+steps_per_episode]) for i in range(0, len(errors), steps_per_episode)]
    
        # Compute rolling average of episode-wise errors
        rolling_avg = np.convolve(episode_errors, np.ones(window_size)/window_size, mode='valid')

        plt.figure(figsize=(10, 5))  # Set the figure size
        plt.plot(rolling_avg, color='blue', lw=2)  # Plot rolling average with blue color and line width of 2
        plt.title('Rolling Average of TD Errors Over Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Error')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_rewards(self, rewards):
        # Cumulative rewards
        cumulative_rewards = np.cumsum(rewards)
    
        # Average rewards per episode
        average_rewards = cumulative_rewards / (np.arange(len(rewards)) + 1)
    
        plt.figure(figsize=(10, 5))
    
        # Plotting both cumulative and average rewards
        plt.plot(cumulative_rewards, label='Cumulative Rewards', color='blue')
        plt.plot(average_rewards, label='Average Rewards per Episode', color='red', linestyle='dashed')
    
        plt.title('Rewards Over Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    DECAY_RATE = 0.89
    MIN_EPSILON = 0.1
    GRID_WIDTH = 7
    GRID_HEIGHT = 4
    NUM_EPISODES = 15000
    catches = 0
    losses = 0
    total_reward = 0
    game = GameOfCatch(GRID_WIDTH, GRID_HEIGHT, NUM_EPISODES)
    visualizer = GameVisualizer(GRID_WIDTH, GRID_HEIGHT)
    rwd_plot = []
    # Training loop: Run for a fixed number of episodes
    for episode in range(NUM_EPISODES):
        # Set the ball at a random column at the top
        ball_row = 0
        ball_col = np.random.randint(game.GRID_WIDTH)
        game.env[:-1, :] = 0
        game.env[0, ball_col] = 1

        # For the basket, if it's the first episode, position it randomly; else, use its last position
        if episode == 0:
            basket_col = np.random.randint(game.GRID_WIDTH)
        else:
            _, _, basket_col = state  # Use the basket's last position from the previous episode

        state = (ball_row, ball_col, basket_col)

        # For each step in the game until the ball reaches the last row
        ball_dropped = False
        while not ball_dropped:
            action = game.choose_action(state, game.epsilon)
            new_state, reward = game.update_environment(game.env, *state, action)
            # Update Q-values
            game.q_learning(state, action, reward, new_state)

            # Only visualize every 100th episode
            if episode % 300 == 0:
                visualizer.visualize(game.env, catches, losses, total_reward)
            
            if new_state[0] == game.GRID_HEIGHT - 1:
                ball_dropped = True  # Signal that this episode is over
            
            state = new_state

        # Update catches, losses, and total reward based on the episode result
        if reward > 0:  # Assumes that a positive reward means a catch
            catches += 1
        else:  # Assumes that a negative or zero reward means a loss
            losses += 1
        total_reward += reward
        rwd_plot.append(total_reward)
        # Only visualize the end state of every 100th episode
        if episode % 300 == 0:
            visualizer.visualize(game.env, catches, losses, total_reward)

        # Decay epsilon after each episode
        game.epsilon = max(MIN_EPSILON, game.epsilon * DECAY_RATE)

    # At the end of training, plot errors
    game.plot_errors(game.errors, window_size=500)
    game.plot_rewards(rwd_plot)  # Assuming total_rewards is a list storing reward for each episode
 