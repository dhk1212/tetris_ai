# Resources / Sources used:
# https://www.youtube.com/watch?v=os4DcbpL0Nc (last accessed 10.02.2024 / 20:48)
# https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/train.py (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=L8ypSXwyBds&t=2145s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=nF_crEtmpBo&t=5s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=t3fbETsIBCY (last accessed 10.02.2024 / 20:48)

import numpy as np

from tetris.agent import Agent
from tetris.game import Tetris

# Initialize the Tetris environment with specified board dimensions
tetris = Tetris(width=10, height=20)

# Define training parameters
max_episodes = 3000
max_steps_per_episode = 25000

# Initialize the agent with the state size of the environment
agent = Agent(state_size=tetris.state_size)

# Lists to track the progress of episodes and rewards
episode_list = []
reward_list = []

for episode_num in range(max_episodes):
    current_state = tetris.reset_game_state()
    is_done = False
    step_count = 0
    episode_reward = 0

    while not is_done and step_count < max_steps_per_episode:
        # Visualize the game state for each step
        tetris.display_game_state(score=episode_reward)

        # Generate all possible states based on the current game state
        possible_next_states = tetris.generate_all_possible_states()

        # Check if no next states are available, indicating the game is over
        if not possible_next_states:
            break

        # Select the best action (state) according to the agent
        chosen_state = agent.choose_action(list(possible_next_states.values()))

        # Find the action (position and rotation) that corresponds to the chosen state
        chosen_action = None
        for action, state in possible_next_states.items():
            if np.array_equal(chosen_state, state):
                chosen_action = action
                break

        # Process the chosen action in the environment and receive the reward
        reward, is_done = tetris.process_action(chosen_action)
        episode_reward += reward

        # Store the experience in the agent's memory for future training
        agent.save(current_state, possible_next_states[chosen_action], reward, is_done)

        # Update the current state for the next iteration
        current_state = possible_next_states[chosen_action]

        step_count += 1

    print(f"Total reward for episode {episode_num}: {episode_reward}")
    episode_list.append(episode_num)
    reward_list.append(episode_reward)

    # Train the agent on a batch of experiences from its memory
    agent.train_on_batch_from_memory()

    # Update the exploration rate (epsilon) of the agent
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay
