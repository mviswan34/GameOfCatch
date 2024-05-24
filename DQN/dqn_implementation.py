import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import random
import pygame
from collections import deque
import matplotlib.pyplot as plt
import time
import pdb

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
        self.basket_image = pygame.image.load("C:/Users/megha/A Thesis/Flappy Bird/Game_Of_catch/basket.png")
        self.basket_image = pygame.transform.scale(self.basket_image, (self.CELL_SIZE, self.CELL_SIZE))
        self.bg_img = pygame.image.load("C:/Users/megha/A Thesis/Flappy Bird/Game_Of_catch/background.png")
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
            self.screen.blit(text, (self.GRID_WIDTH * self.CELL_SIZE - 200, 20 + index * 40))  # Adjust the position based on your preference
    
        pygame.display.flip()  # Update the screen
        pygame.time.wait(self.delay_time)  # Introduce delay to control speed

class CatchGame:
    def __init__(self, width=7, height=4):
        self.width = width
        self.height = height
        self.visualizer = GameVisualizer(width, height)
        # Initialize the ball's position randomly
        self.ball_column = np.random.choice(self.width)
        # Initialize the agent's position randomly for the very first episode
        self.agent_column = np.random.choice(self.width)
        self.state = self.reset(first_reset=True)

    def reset(self, first_reset=False):
        self.state = np.zeros((self.height, self.width))
        # Place the ball at a random column in the top row
        self.ball_column = np.random.choice(self.width)
        self.state[0, self.ball_column] = 1

        # If it's not the first reset, agent retains its position from previous episode
        if not first_reset:
            self.state[self.height-1, self.agent_column] = 2
        else:
            # Initialize agent at a random column in the bottom row only for the first episode
            self.agent_column = np.random.choice(self.width)
            self.state[self.height-1, self.agent_column] = 2
        return self.state


    def step(self, action):
        # 1. Move the agent based on the action
        if action == 0:  # Move Left
            self.agent_column = max(0, self.agent_column - 1)
        elif action == 1:  # Move Right
            self.agent_column = min(self.width - 1, self.agent_column + 1)
        # If action == 2, the agent stays in place, so no need to update self.agent_column

        # 2. Move the ball down by one cell
        for i in range(self.height - 2, -1, -1):
            if 1 in self.state[i]:  # If there's a ball in the current row
                ball_col_idx = np.where(self.state[i] == 1)[0][0]  # Get the column index of the ball
                self.state[i + 1, ball_col_idx] = 1  # Move the ball one row down
                self.state[i, ball_col_idx] = 0  # Set the current cell to empty

        # 3. Calculate the reward based on agent's and ball's positions
        reward = -0.1 if action in [0, 1] else 0  # Penalty for moving left or right
        if 1 in self.state[self.height - 1]:  # If ball is in the bottom row
            if self.state[self.height - 1, self.agent_column] == 1:  # If ball is in the same column as the agent
                reward += 10
            else:
                reward -= 10

        # 4. Determine if the episode has ended
        done = True if 1 in self.state[self.height - 1] else False

        # Update the agent's position in the state
        self.state[self.height - 1, :] = 0  # Clear the bottom row
        self.state[self.height - 1, self.agent_column] = 2  # Place the agent in its new column

        return np.copy(self.state), reward, done


    def render(self, catches, losses, total_reward):
        self.visualizer.visualize(self.state, catches, losses, total_reward)


class DQNAgent:
    def __init__(self, input_shape, action_space):
        self.input_shape = input_shape
        self.action_space = action_space
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.main_model = self._build_model()
        self.target_model = self._build_model()
        self.memory = deque(maxlen=2000)  # set a max length e.g., 2000
        self.td_errors = []

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(optimizer=Adam(), loss='huber')
        return model
    
    """
        potential adjustments, should you face issues:

        Increase or Decrease Complexity: If your agent isn't learning well, you could try increasing the complexity (adding more neurons or layers).
        Conversely, if training is unstable, you might benefit from a simpler model.

        Regularization: If you suspect overfitting, consider adding dropout layers or L2 regularization.

        Learning Rate: Adjusting the learning rate of the Adam optimizer can sometimes have a significant impact on training dynamics.

        Alternative Architectures: While the feedforward design is a solid start, you might also experiment with architectures like Convolutional Neural Networks (CNNs), 
        especially if the grid size becomes much larger or the game's visual patterns become more intricate.
    """

    def remember(self, state, action, reward, next_state, done):
        # Experience Replay: Rather than learning immediately from the most recent experience, the agent stores these experiences in memory.
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        #Epsilon-Greedy strategy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.main_model.predict(state)
        return np.argmax(q_values[0])



    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        pdb.set_trace()

        states = np.squeeze(np.array([i[0] for i in minibatch]))
        next_states = np.squeeze(np.array([i[3] for i in minibatch]))

        # Predict Q-values for current and next states
        current_qs = self.main_model.predict(states)
        next_qs = self.target_model.predict(next_states)

        X = []
        y = []

        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                new_q = reward
            else:
                new_q = reward + self.gamma * np.amax(next_qs[index])

            # Compute the TD error
            td_error = new_q - current_qs[index][action]
            self.td_errors.append(td_error)  # Storing the TD error

            # Update Q value for given state
            current_q_values = current_qs[index]
            current_q_values[action] = new_q

            X.append(np.squeeze(state))
            y.append(current_q_values)
        """
        Do a forward pass,calculate the loss 
        """
        self.main_model.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=0, shuffle=False)
        # Decay function with controlled start and end
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def update_target_network(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def load(self, name):
        self.main_model.load_weights(name)
        self.update_target_network()

    def save(self, name):
        self.main_model.save_weights(name)

def moving_average(data, window_size):
    return [np.mean(data[max(0, i-window_size):(i+1)]) for i in range(len(data))]


if __name__ == "__main__":
    start_time = time.time()
    env = CatchGame()
    agent = DQNAgent(env.state.shape, 3)  # 3 actions: left, right, stay
    episodes = 5000
    TRAIN_FREQUENCY = 15
    catches, losses, total_reward = 0, 0, 0
    episode_rewards = []
    for e in range(episodes):
        #print("Episode: ", e)
        state = env.reset()
        state = np.reshape(state, [1, env.height, env.width])
        r_track = 0
        for time_instance in range(4):  # 4 time instances per episode as per your design
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.height, env.width])
            agent.remember(state, action, reward, next_state, done)
            #print("Reward: ",reward)
            state = next_state
            total_reward += reward
            r_track += reward
            if reward > 0: 
                catches += 1
            elif reward < -1:
                losses += 1
            if e % TRAIN_FREQUENCY == 0:
                agent.train(128)  # Train with batch_size=32 (arbitrary choice)
            if e % 100 == 0:
                env.render(catches, losses, round(total_reward, 2))
            if done:
                break
        episode_rewards.append(r_track)
        #print("Catches: ",catches)
        #print("Losses: ",losses)
        if e % 100 == 0:  # Every 100 episodes, update target network
            agent.update_target_network()
        print("Episode: ", e)
        print("Reward: ",total_reward)
    smoothed_rewards = moving_average(episode_rewards, window_size=10)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training took {duration:.2f} seconds")

    avg_td_errors_per_episode = [np.mean(agent.td_errors[i:i+4]) for i in range(0, len(agent.td_errors), 4)]

    # Plotting TD errors
    plt.figure(figsize=(10, 5))
    plt.plot(avg_td_errors_per_episode)
    plt.title('TD Errors over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('TD Error')
    plt.show()


    # Plotting Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_rewards)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()