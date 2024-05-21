import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import time
import altair as alt


########################################################################## Q LEARNING #############################################################
class GameOfCatch:
    # Initialize the game parameters and sets up the environment.
    def __init__(self, GRID_WIDTH, GRID_HEIGHT, NUM_EPISODES, ALPHA, GAMMA, EPSILON):
        self.NUM_EPISODES = NUM_EPISODES
        self.GRID_WIDTH = GRID_WIDTH
        self.GRID_HEIGHT = GRID_HEIGHT
        self.env = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        self.epsilon = EPSILON
        self.q_table = np.zeros((GRID_HEIGHT, GRID_WIDTH, GRID_WIDTH, 3))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.errors = []

    # Update the environment based on the ball and basket positions and the action taken.
    def update_environment(self, env, ball_row, ball_col, basket_col, action, prob):
        #print(f"Probabilities: {prob}")
        movement_penalty = -0.01
        movement_options = ["left", "down", "right"]
        ball_movement = np.random.choice(movement_options, p=prob)
        #print(f"Ball movement selected: {ball_movement}")
        
        # Ball Movements
        if ball_movement == "left":
            ball_col = max(0, ball_col - 1)
        elif ball_movement == "right":
            ball_col = min(self.GRID_WIDTH - 1, ball_col + 1)
        ball_row = min(self.GRID_HEIGHT - 1, ball_row + 1)  # Always move down
        #print(f"Updated ball position: row={ball_row}, col={ball_col}")
        
        # Basket Movements
        if action == 0:  # Move left
            if basket_col > 0:
                basket_col -= 1
                reward = movement_penalty
            else:
                reward = 0  
        elif action == 1:  # Move right
            if basket_col < self.GRID_WIDTH - 1:
                basket_col += 1
                reward = movement_penalty
            else:
                reward = 0 
        elif action == 2:  # Stay
            reward = 0

        if ball_row == self.GRID_HEIGHT - 1:
            if ball_col == basket_col:
                reward += 1.0
            else:
                reward -= 1.0

        env.fill(0)
        env[ball_row, ball_col] = 1
        env[self.GRID_HEIGHT - 1, basket_col] = 2

        new_state = (ball_row, ball_col, basket_col)
        return new_state, reward


    # Choose an action based on the current state and exploration rate.
    def choose_action(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice([0, 1, 2])
        else:
            ball_row, ball_col, basket_col = state
            action = np.argmax(self.q_table[ball_row, ball_col, basket_col, :])
        return action

    # Implement the Q-learning algorithm, updating the Q-table based on the action taken and the reward received.
    def q_learning(self, state, action, reward, new_state):
        current_q = self.q_table[state[0], state[1], state[2], action]
        expected_q = np.max(self.q_table[new_state[0], new_state[1], new_state[2], :])
        td = reward + self.gamma * expected_q - current_q
        self.q_table[state[0], state[1], state[2], action] += self.alpha * td
        self.errors.append(td)
        return td

### Plot the game for visualization purpose
def plot_game_environment(state, grid_height, grid_width, reward=None):
    ball_row, ball_col, basket_col = state
    plt.figure(figsize=(5, 5), dpi=100)

    # Load and display background image
    background_image_path = "images/inverted_bg.png"
    background = plt.imread(background_image_path)
    plt.imshow(background, extent=[0, grid_width, grid_height, 0])

    # Set the axis limits
    plt.xlim(0, grid_width)
    plt.ylim(0, grid_height)

    # Create a red circle for the ball
    ball_x = ball_col + 0.5
    ball_y = grid_height - ball_row - 0.5
    ball_circle = patches.Circle((ball_x, ball_y), radius=0.2, facecolor='red')
    plt.gca().add_patch(ball_circle)

    # Create a brown rectangle for the basket
    basket_width = 0.6
    basket_height = 0.3
    basket_color = 'brown'
    # Calculate the lower left corner of the basket to center it in the cell
    basket_x = basket_col + 0.5
    basket_y = 0.5 
    basket = patches.Rectangle((basket_x- basket_width / 2, basket_y- basket_height / 2), basket_width, basket_height, color=basket_color)
    plt.gca().add_patch(basket)

    # Add grid lines for better visualization
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    return buf.getvalue()

def visualize_and_train(game, prob, visualize, visualize_every_n_episodes=1000):
    catches = 0
    losses = 0
    DECAY_RATE = 0.995
    MIN_EPSILON = 0.1

    game_placeholder = st.empty()

    # Initializing placeholders and columns
    if visualize == "True":
        ss1, ss2 = st.columns([1,3])
        with ss1:
            game_placeholder = st.empty()

        with ss2:
            st.markdown("<h4 style='text-align: center;'>Total Rewards over Time</h4>", unsafe_allow_html=True)
            reward_chart_placeholder = st.empty()


        ss3,ss4 = st.columns([1,3])
        with ss3:
            st.markdown(" ")
            st.markdown(" ")
            st.markdown(" ")
            st.markdown(" ")
            num_catches_placeholder = st.empty()
            num_losses_placeholder = st.empty()
            total_value_placeholder = st.empty()
        with ss4:
            st.markdown("<h4 style='text-align: center;'>TD Errors over Time</h4>", unsafe_allow_html=True)
            td_error_chart_placeholder = st.empty()
    else:
        ss1, ss2 = st.columns([1,3])
        with ss1:
            st.markdown(" ")
            st.markdown(" ")
        
            num_catches_placeholder = st.empty()
            num_losses_placeholder = st.empty()
            total_value_placeholder = st.empty()

        with ss2:
            st.markdown("<h4 style='text-align: center;'>Total Rewards over Time</h4>", unsafe_allow_html=True)
            reward_chart_placeholder = st.empty()


        ss3,ss4 = st.columns([0.1,3])
        with ss3:
            st.markdown(" ")
            
        with ss4:
            st.markdown("<h4 style='text-align: center;'>TD Errors over Time</h4>", unsafe_allow_html=True)
            td_error_chart_placeholder = st.empty()

    game_reward = 0
    episode_rewards = []
    episode_td_errors = []
    
    reward_df = pd.DataFrame(data=[], columns=['Episode', 'Total Reward'])
    td_error_df = pd.DataFrame(data=[], columns=['Episode', 'TD Error'])

    total = 0.0

    for episode in range(game.NUM_EPISODES):
        ball_row = 0
        ball_col = np.random.randint(game.GRID_WIDTH)
        game.env[:-1, :] = 0
        game.env[0, ball_col] = 1

        if episode == 0:
            basket_col = np.random.randint(game.GRID_WIDTH)
        else:
            _, _, basket_col = state

        state = (ball_row, ball_col, basket_col)
        ball_dropped = False
        epi_reward = 0
        episode_td_error_sum = 0
        episode_td_error_count = 0
        ep_rewards = []

        while not ball_dropped:
            action = game.choose_action(state, game.epsilon)
            new_state, reward = game.update_environment(game.env, *state, action, prob)
            td_error = game.q_learning(state, action, reward, new_state)
            episode_td_error_sum += abs(td_error)
            episode_td_error_count += 1
            epi_reward += reward
            state = new_state
            ep_rewards.append(reward)

            if visualize == "True" and episode % visualize_every_n_episodes == 0:
                with ss1:
                    image1 = plot_game_environment(state, game.GRID_HEIGHT, game.GRID_WIDTH, epi_reward)
                    game_placeholder.image(image1, caption=f"Episode: {episode}", width=400)
                    time.sleep(0.15)
                with ss3:
                    num_catches_placeholder.markdown(f'Number of Catches = {catches}')
                    num_losses_placeholder.markdown(f'Number of Losses = {losses}')
                    total_value_placeholder.markdown(f'Reward = {total:.2f}')

            if new_state[0] == game.GRID_HEIGHT - 1:
                ball_dropped = True

        game_reward += epi_reward
        episode_rewards.append(game_reward)
        episode_td_errors.append(episode_td_error_sum / episode_td_error_count)

        if reward > 0:
            catches += 1
        else:
            losses += 1

        game.epsilon = max(MIN_EPSILON, game.epsilon * DECAY_RATE)

        # Update the new dataframes
        last_index = reward_df.index.max() 
        if pd.isna(last_index):
            last_index = -1
        new_index = last_index + 1
        total += sum(ep_rewards)
        reward_df.loc[new_index] = [episode, total]
        td_error_df.loc[new_index] = [episode, td_error]

        if episode % visualize_every_n_episodes == 0:
            if visualize != "True":
                with ss1:
                    num_catches_placeholder.markdown(f'Number of Catches = {catches}')
                    num_losses_placeholder.markdown(f'Number of Losses = {losses}')
                    total_value_placeholder.markdown(f'Reward = {total:.2f}')
            with ss2:
                    reward_chart_placeholder.line_chart(reward_df, x='Episode', y='Total Reward')
            with ss4:
                    td_error_chart_placeholder.line_chart(td_error_df, x='Episode', y='TD Error', use_container_width=True)


    st.markdown("Training complete!")
    st.markdown(f"Number of Catches: {catches}")
    st.markdown(f"Number of Losses: {losses}")
    st.markdown(f"Game Total Reward = {game_reward:.2f}")
    return game  # Return the trained game object


### Test the trained agent.
def test_game_of_catch(game, num_throws, prob):
    test_catches = 0
    test_losses = 0
    game_placeholder = st.empty()

    for i in range(num_throws):
        ball_row = 0
        ball_col = np.random.randint(game.GRID_WIDTH)
        basket_col = np.random.randint(game.GRID_WIDTH)
        state = (ball_row, ball_col, basket_col)

        ball_dropped = False
        while not ball_dropped:
            action = game.choose_action(state, 0)  # Epsilon set to 0 for testing
            new_state, reward = game.update_environment(game.env, *state, action, prob)
            state = new_state
            image1 = plot_game_environment(state, game.GRID_HEIGHT, game.GRID_WIDTH, reward)
            game_placeholder.image(image1, caption=f"Episode: {i}", width=400)
            time.sleep(0.2)

            if new_state[0] == game.GRID_HEIGHT - 1:
                ball_dropped = True
                if reward > 0:
                    test_catches += 1
                else:
                    test_losses += 1

    st.markdown(f"Testing complete! Catches: {test_catches}, Losses: {test_losses}")

### Streamlit Session State Management
def show_config():
    # Streamlit applications are interactive and stateful. 
    # However, by default, each interaction with the user interface (like a button click) causes the whole script to rerun.
    # This will reset all variables, therefore Streamlit has given session state.
    # 'show_config' is a flag used to track the configuration within the application.
    st.session_state.show_config = True

# Define a function to handle button clicks and set the session state
def set_visualize(visualize):
    st.session_state.visualize = visualize

### Main Streamlit app function
def main():
    st.set_page_config(layout="wide")
    
    #HTML Font Description
    st.markdown('<style>h1 { font-size: 32px; text-align: center; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p1 { font-size: 25px; text-align: justify; }</style>', unsafe_allow_html=True)
    st.markdown('<style>h2 { font-size: 25px; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p2 { font-size: 22px; }</style>', unsafe_allow_html=True)
    st.markdown('<style>.custom-font { font-family: "Verdana, sans-serif"; font-size: 20px; }</style>', unsafe_allow_html=True)
    ########           st.markdown('<p1></p1>', unsafe_allow_html=True)
    #Title
    st.title('Code & Catch Quest: Programming AI for Strategic Play')
    st.markdown(""" <div style="text-align: center;"><h2>AUTHOR: MEGHA VISWANATH</h2></div>""",unsafe_allow_html=True)
    
    # Set the show_config in the session state, so that its value will persist across reruns of the script within the same session. 
    if 'show_config' not in st.session_state:
        st.session_state.show_config = False

    st.write("Current working directory:", os.getcwd())

    #Introduction
    st.markdown("""<div style='text-align: justify; font-size: 25px;'>
                        Welcome to 'Code & Catch Quest: Programming AI for Strategic Play.' This site offers a clear view into the world of 
                artificial intelligence as it tackles the realm of games. It's a place to see firsthand how AI can be taught to think, decide, 
                and learn through Q-Learning and Deep Q-Learning. The journey begins with a simple game that serves as a testing ground for 
                these concepts and extends into the application within a two-player zero-sum stochastic game. For anyone intrigued by AI's 
                capabilities, this blog aims to present these complex ideas in an accessible format.<br/><br/><br/>
                    </div>""", unsafe_allow_html=True)
    
    # The Game
    st.divider()
    c2,c3,c4 = st.columns([3,0.25,1])
    with c2:
        st.markdown("<h1>The Game</h1>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: justify; font-size: 25px;'>
                    Catch is a basic arcade game where you catch falling objects like fruits or eggs with a basket. 
                    You score a point for every catch and lose one for each miss. In this exploration, we'll train a computer to 
                    play Catch by itself. However, instead of the colorful game shown on the right, we use a simpler version for ease 
                    of understanding, which you'll see shortly.<br/>
                    In our version, the AI agent chooses to move the basket left, right, or stay in place based on where the fruit is falling. 
                    The challenge is to build a model that learns the best action to take at each moment to maximize the score. 
                    This learning process doesn't rely on data from expert players but is achieved through the model's own experience, 
                    much like how humans learn new games.<br/>
                    An extra twist in our game is the stochastic movement of the falling object. This means that as the object falls, 
                    its path can change unpredictably, sometimes moving diagonally left, right, or straight down. This stochastic component 
                    adds an element of unpredictability, testing the model's ability to adapt and make decisions in a situation with 
                    inherent uncertainty.
                    </div>""", unsafe_allow_html=True)
    
    with c4:
        image_path = "https://github.com/mviswan34/GameOfCatch/blob/main/WebApp/images/small.gif"
        st.image(image_path, caption="Credits:@vsgif.com")

    st.divider()

    # Q Learning
    st.markdown("<h1>Q LEARNING</h1>", unsafe_allow_html=True)
    
    ### What is Q Learning
    c5,c6,c7 = st.columns([2.4,0.1,2])
    with c5:
        st.markdown("""<div style='text-align: justify; font-size: 23.5px;'>
                    <b>What is Q-Learning?</b></div>""", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: justify; font-size: 21.5px;'>
                Q-learning is a method that helps an AI agent learn the best moves through practice, aiming for the highest rewards from its 
                    actions. The "Q" stands for "quality," assessing how each move contributes to the ultimate goal of winning the game. 
                    In our game of Catch, Q-learning is like a coach for the AI agent, directing it on the most successful strategies to 
                    catch objects and rack up points.
                </div>""", unsafe_allow_html=True)
    with c7:
        
        st.markdown("""<style>
                    /* Targeting the container of the radio option text */
                    div[data-testid="stMarkdownContainer"] > 
                    p {font-size: 23.5px !important;}
                    </style>
                    """, unsafe_allow_html=True)
        property = st.radio("**What are the model properties?**", ["Model-Free", "Value-Based", "Off-Policy"], horizontal = True)

        if property == "Model-Free":
            st.markdown("""<div style='text-align: justify; font-size: 21.5px;'>The agent doesn't know the rules of the game in advance. 
                        It learns by playing - catching objects earning points, missing them losing points.</div>""", unsafe_allow_html=True)
        elif property == "Value-Based":
            st.markdown("""<div style='text-align: justify; font-size: 21.5px;'>Each possible move (left, right, stay) 
                        gets a score based on its potential to earn more points in the future. The agent learns to choose moves with 
                        higher scores.</div>""", unsafe_allow_html=True)
        elif property == "Off-Policy":
            st.markdown("""<div style='text-align: justify; font-size: 21.5px;'>The agent can learn from the success of other agents' strategies, 
                        not just its own.</div>""", unsafe_allow_html=True)
    
    ### How does Q-Learning Work?
    st.markdown("""<div style='text-align: justify; font-size: 25px;'>
                <br/><b>How does Q Learning Work?</b><br/></div>""", unsafe_allow_html=True)
    c8,c9,c10 = st.columns([1.4,0.01,3])
    with c8:
        t1,t2 = st.columns([0.1,1])
        with t1:
            st.markdown(" ")
            arrow = "images/Arrow_right.png"
            st.image(arrow)
        with t2:
            st.markdown("""<div style='text-align: justify; font-size: 23.5px;'>
                <b>Q Table</b></div>""", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: justify; font-size: 21.5px;'>
                    In the Game of Catch, our AI agent uses what's known as a Q-table to guide its decisions. 
                        The Q-table is a grid of values representing all possible states (positions of the falling objects) 
                        and actions (movements of the basket). Initially, this table is filled with zeroes, but as the game progresses, 
                        the agent updates these values based on its experiences — catches and misses.<br/>
                        </div>""", unsafe_allow_html=True)
    with c10:
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        qtable_img = "images/q_table.png"
        st.image(qtable_img)
    
    st.markdown(" ")
    
    
    t1,t2 = st.columns([0.03,1])
    with t1:
            st.markdown(" ")
            arrow = "images/Arrow_right.png"
            st.image(arrow)
    with t2:
            st.markdown("""<div style='text-align: justify; font-size: 23.5px;'>
                <b>Q Function</b></div>""", unsafe_allow_html=True) 
    st.markdown("""
            <div style='text-align: justify; font-size: 21.5px;'>
                Each entry in the Q-table represents the "Q-value," of taking a specific action in a particular state. 
                This value is updated using the Q-function, which combines immediate rewards with anticipated future rewards. 
                The process involves the following steps:<br/>
                </div>
            """, unsafe_allow_html=True)
                
        
    c11,c12,c13 = st.columns([2,0.01,1])  
    with c11:
            st.markdown("""
            <div style='text-align: justify; font-size: 21.5px;'>
                        1. Start with a Q-table filled with zeroes, representing an inexperienced agent.<br/>
                2. The agent selects an action based on either exploration (trying new things) or exploitation (using known information).<br/>
                3. The agent acts, and the environment provides feedback in the form of rewards or penalties.<br/>
            4. The Q-value for the taken action is recalculated with the Q-Function given on the left:
                        <ul style="list-style-type: square;">
                    <li style="font-size: 22px;">Take the existing Q-value (knowledge before this move).</li>
                    <li style="font-size: 22px;">Add a portion (determined by the learning rate, α) of:
                        <ul style="list-style-type: circle;">
                            <li style="font-size: 22px;">The immediate reward received.</li>
                            <li style="font-size: 22px;">Plus the discounted (by a factor, γ) best Q-value for the next state (what the agent expects to 
                            gain in the future).</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)      
    with c13:
            st.markdown(" ")
            st.markdown(" ")
            qtable_img = "images/Q_Equation.png"
            st.image(qtable_img)
    st.markdown("""
            <div style='text-align: justify; font-size: 21.5px;'>
                This updated Q-value reflects the agent's improved understanding after considering both the immediate result and the 
                potential future benefits.
                By repeating these steps, the Q-learning algorithm helps the agent learn the optimal policy, guiding it toward actions that 
                maximize rewards over time.
            </div>
            """, unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(" ")
    
    c14,c15,c16 = st.columns([0.2,0.8,0.2])
    with c15:
        q_algo = "images/q_algorithm.png"
        st.image(q_algo, caption="Q Algorithm")  

    st.divider()
    ### Test the algorithm
    st.markdown("<h1>TRAIN THE AGENT</h1>", unsafe_allow_html=True)
    st.markdown(" ")

    c17,c18,c19 = st.columns([1,0.2,1])
    with c17:
        st.markdown("""
            <div style='text-align: justify; font-size: 21.5px;'>
                Let us start by outlining the game's playing field. Set the size of the grid:
            </div>
            """, unsafe_allow_html=True)
        st.markdown(" ")
        cc1,cc2 = st.columns([1,1])
        with cc1:
                #st.markdown('<p2 class="custom-font">Select the Width:</p2>', unsafe_allow_html=True)
                GRID_WIDTH = st.slider("Width: ",4,20,5)

        with cc2:
                #st.markdown('<p2 class="custom-font">Select the Height:</p2>', unsafe_allow_html=True)
                GRID_HEIGHT = st.slider("Height",4,20,5)
    with c19:
        st.markdown("""
            <div style='text-align: justify; font-size: 21.5px;'>
                    Stochastic Component:
                    Set the ball's probability to fall straight downwards:
            </div>
            """, unsafe_allow_html=True)
        st.markdown(" ")
        cc3, cc4, cc5 = st.columns([1,0.2, 1])
        with cc3:
            #st.markdown('<p2 class="custom-font">Probability (Straight Downwards):</p2>', unsafe_allow_html=True)
            prob_down = st.number_input("Probability (Straight Downwards):", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

        prob_left = (1 - prob_down) / 2
        prob_right = (1 - prob_down) / 2

        with cc5:
            st.markdown(f'<p2 class="custom-font">Probability (Diagonally Left): {prob_left:.2f}</p2>', unsafe_allow_html=True)
            st.markdown(f'<p2 class="custom-font">Probability (Diagonally Right): {prob_right:.2f}</p2>', unsafe_allow_html=True)

        prob = [prob_left, prob_down, prob_right]

        
    
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    x1,x2 =st.columns([3.6,1])
    with x1:
        st.markdown("""
            <div style='text-align: justify; font-size: 21.5px;'>
                Would you like to tailor the algorithm's settings to your preference? If so, click on the configure button, or else we shall proceed with their default values.
            </div>""", unsafe_allow_html=True)
    with x2:
        st.button("Configure settings", on_click=show_config)

    st.markdown(" ")
    if st.session_state.show_config:
        cc6,mid1,cc7,mid2,cc8,mid3,cc9  = st.columns([0.8, 0.1, 1.1, 0.1, 0.9, 0.1, 1.1])
        with cc6:
            st.markdown("#### Number of Episodes")
            st.markdown("""<div style='text-align: justify; font-size: 18px;'>
                        This determines the total number of games the Q-learning algorithm will play to learn the optimal strategy.
                        </div>""", unsafe_allow_html=True)
            NUM_EPISODES = st.slider('N = ', min_value=1, max_value=50000, value=12000, step=1)
            NUM_EPISODES = NUM_EPISODES+1
        with cc7:
            st.markdown("#### Epsilon (Exploration Rate)")
            st.markdown("""<div style='text-align: justify; font-size: 18px;'>
                        The probability that our agent will explore (i.e., choose a random action). A higher value promotes more exploration 
                        at the cost of possibly making more mistakes.
                        </div>""", unsafe_allow_html=True)
            EPSILON = st.slider('ε = ', min_value=0.0, max_value=1.0, value=0.85, step=0.01)
        with cc8:
            st.markdown("#### Alpha (Learning Rate)")
            st.markdown("""<div style='text-align: justify; font-size: 18px;'>
                        Controls how much of the new Q-value estimate we'll adopt. A higher value makes our algorithm learn faster 
                        but can also make it unstable.
                        </div>""", unsafe_allow_html=True)
            ALPHA = st.slider('α = ', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        with cc9:
            st.markdown("#### Gamma (Discount Factor)")
            st.markdown("""<div style='text-align: justify; font-size: 18px;'>
                        This value determines how much future rewards are valued compared to immediate rewards. 
                        A higher value promotes considering future rewards over immediate ones.
                        </div>""", unsafe_allow_html=True)
            GAMMA = st.slider('γ = ', min_value=0.0, max_value=1.0, value=0.99, step=0.01)
    else:
        NUM_EPISODES = 12001
        EPSILON = 0.85
        ALPHA = 0.1
        GAMMA = 0.99

    game = GameOfCatch(GRID_WIDTH, GRID_HEIGHT, NUM_EPISODES, ALPHA, GAMMA, EPSILON)

    # Initialize trained_agent in session state if it's not already there
    if 'trained_agent' not in st.session_state:
        st.session_state.trained_agent = None
    st.markdown(" ")
    st.markdown(" ")
    ccc1, ccc2, ccc3 = st.columns([0.5,0.08,0.8])
    with ccc1:
        st.markdown("<h4 style='text-align: justify;'>Would you like to visualize some episodes during training?</h4>", unsafe_allow_html=True)

    # Visualize training decision buttons
    viz=3
    with ccc2:
        if st.button("Yes"):
            viz = 1
            
    with ccc3:
        if st.button("No"):
            viz=0
    if viz==1:
            st.session_state.trained_agent = visualize_and_train(game, prob, visualize="True")
    elif viz==0:
            st.session_state.trained_agent = visualize_and_train(game, prob, visualize="False")

    st.divider()

    ### Test the algorithm
    st.markdown("<h1>TEST THE AGENT</h1>", unsafe_allow_html=True)
    st.markdown("""
            <div style='text-align: justify; font-size: 21.5px;'>
                On how many episodes would you like to test the agent?
            </div>""", unsafe_allow_html=True)

    TEST_EPISODES = st.slider('N = ', min_value=1, max_value=100, value=20, step=1)

    if st.button("Test your agent"):
        if st.session_state.trained_agent is not None:
            test_game_of_catch(st.session_state.trained_agent, TEST_EPISODES, prob)
        else:
            st.error("Please train your agent before testing.")



if __name__ == "__main__":
    main()
