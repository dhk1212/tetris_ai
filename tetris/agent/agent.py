# Resources / Sources used:
# https://www.youtube.com/watch?v=os4DcbpL0Nc (last accessed 10.02.2024 / 20:48)
# https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/agent.py (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=L8ypSXwyBds&t=2145s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=nF_crEtmpBo&t=5s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=t3fbETsIBCY (last accessed 10.02.2024 / 20:48)

import random
import sys
from collections import deque
from typing import List, Optional

import keras
import numpy as np
from keras.layers import Dense


class Agent:
    # Configuration
    MEMORY_SIZE = 30000
    DISCOUNT = 0.95
    EPSILON_START = 1.0
    EPSILON_MIN = 0.001
    EPSILON_END_EPISODE = 2000
    BATCH_SIZE = 512
    REPLAY_START = 3000
    EPOCHS = 1

    def __init__(self, state_size):
        # Initialise values according to configuration
        self.state_size = state_size
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.discount = self.DISCOUNT
        self.epsilon = self.EPSILON_START
        self.epsilon_min = self.EPSILON_MIN
        self.epsilon_end_episode = self.EPSILON_END_EPISODE
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_end_episode
        self.batch_size = self.BATCH_SIZE
        self.replay_start = self.REPLAY_START
        self.epochs = self.EPOCHS

        # Create model
        self.model = self.create_model()

    def create_model(self) -> keras.Sequential:
        """
        Constructs the neural network model for the agent.

        The model is a simple feedforward neural network with ReLU activations
        for hidden layers and a linear activation for the output layer. It is
        designed to approximate the Q-value function for the agent's state space.

        Returns:
            Sequential: The compiled Keras model ready for training.
        """
        model = keras.Sequential([
            Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'),
            Dense(64, activation='relu', kernel_initializer='glorot_uniform'),
            Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
            Dense(1, activation='linear')
        ])

        # Compile the model with mean squared error loss and the Adam optimizer
        model.compile(loss='mse', optimizer='adam')

        return model

    def save(self, current_state: np.ndarray, next_state: np.ndarray, reward: float, done: bool) -> None:
        """
        Stores a transition in the agent's memory.

        This method adds a state transition tuple to the agent's memory, which includes
        the current state, the next state after taking an action, the reward received,
        and a boolean indicating whether the episode has ended (done).

        Parameters:
        - current_state (np.ndarray): The current state of the environment.
        - next_state (np.ndarray): The state of the environment after taking an action.
        - reward (float): The reward received after taking the action.
        - done (bool): A flag indicating whether the episode has terminated.

        Returns:
        - None
        """
        self.memory.append((current_state, next_state, reward, done))

    def choose_action(self, states: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Determines the agent's action from a set of possible states.

        This method decides on the next action by selecting the state with the highest
        predicted value from the model, given the current epsilon-greedy policy. With
        probability epsilon, a random state is chosen (exploration), and with probability
        1 - epsilon, the state with the highest predicted value is chosen (exploitation).

        Parameters:
        - states (List[np.ndarray]): A list of possible next states.

        Returns:
        - Optional[np.ndarray]: The chosen next state. None if states list is empty.
        """
        max_value = -sys.maxsize - 1
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            for state in states:
                value = self.model.predict(np.reshape(state, [1, self.state_size]))[0]
                if value > max_value:
                    max_value = value
                    best_state = state

        return best_state

    def train_on_batch_from_memory(self) -> None:
        """
        Trains the agent's model on a batch of experiences from memory.

        This method implements the experience replay mechanism where the agent
        samples a batch of past - experiences from its memory and updates its model.
        This approach helps to stabilize and improve the learning process.

        The method only proceeds if the memory size exceeds a specified starting threshold,
        ensuring that there's enough data for meaningful training.

        Returns:
        - None
        """
        if len(self.memory) < self.replay_start:
            return  # Not enough data in memory to replay

        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)

        # Extract information from the batch
        states = np.array([experience[0] for experience in batch])
        next_states = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        dones = np.array([experience[3] for experience in batch])

        # Predict Q-values for the next states
        next_q_values = self.model.predict(next_states)

        # Ensure next_q_values is 2-dimensional
        if next_q_values.ndim == 1:
            print(f"next q values are invalid - use fallback and continue learning")
            next_q_values = next_q_values.reshape(-1, 1)

        # Calculate target Q-values
        target_q_values = rewards + self.discount * np.max(next_q_values, axis=1) * (~dones)

        # Train the model
        self.model.fit(states, target_q_values, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
