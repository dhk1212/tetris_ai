# Resources / Sources used:
# https://www.youtube.com/watch?v=os4DcbpL0Nc (last accessed 10.02.2024 / 20:48)
# https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=L8ypSXwyBds&t=2145s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=nF_crEtmpBo&t=5s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=t3fbETsIBCY (last accessed 10.02.2024 / 20:48)
o
import random
from typing import Tuple

import cv2 as cv
import numpy as np

from tetris.board import is_occupied, clear_lines, evaluate_board_state, hard_drop
from .tetrominoes import shapes, shape_names, rotated

# Colors
green = (156, 204, 101)
black = (0, 0, 0)
white = (255, 255, 255)


class Tetris:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=float)

        # State size (Clearede lines, bumpiness, holes, height)
        self.state_size = 4

        # For running the engine
        self.score = -1
        self.anchor = None
        self.shape = None

        # Used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # Reset after initializing
        self.reset_game_state()

    def _select_next_tetromino(self) -> list:
        """
        Selects the next tetromino shape for the game, aiming for a balanced shape distribution.

        This method prioritizes selecting tetromino shapes that have been used less frequently
        to ensure a balanced distribution of shapes over time. If all shapes have been used equally,
        it selects a shape at random.

        Returns:
            list: The coordinates defining the selected tetromino shape.
        """
        # Find the maximum count of any shape being chosen so far
        max_count = max(self._shape_counts)

        # Identify tetromino shapes that have been chosen less than the max count
        valid_tetrominos = [shape_names[i] for i in range(len(shapes)) if self._shape_counts[i] < max_count]

        # If all tetrominos have been chosen equally, select randomly from all shapes
        # Otherwise, select randomly from the less frequently chosen shapes
        if not valid_tetrominos:
            tetromino = random.choice(shape_names)
        else:
            tetromino = random.choice(valid_tetrominos)

        # Increment the count for the chosen tetromino
        self._shape_counts[shape_names.index(tetromino)] += 1

        # Return the coordinates for the selected tetromino shape
        return shapes[tetromino]

    def initialize_new_tetromino(self) -> None:
        """
        Initializes a new tetromino piece for the game.

        This method sets the starting position of the new tetromino piece by placing its anchor
        at the middle top of the game board and selects the next tetromino shape from the available
        shapes ensuring a balanced distribution.
        """
        # Calculate the initial anchor position at the middle top of the game board
        self.anchor = (self.width // 2, 1)  # Use integer division for a discrete grid

        # Select the next tetromino shape based on a balanced distribution strategy
        self.shape = self._select_next_tetromino()

    def is_drop_blocked(self) -> bool:
        """
        Determines if moving the tetromino shape one position down is blocked.

        This method checks if the tetromino shape, when moved one position downward
        from its current anchor point, would collide with the bottom of the game board
        or any existing pieces on the board.

        Returns:
            bool: True if the shape cannot be moved down further without collision; False otherwise.
        """
        # Check for collision one position below the current anchor point
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def process_action(self, action: Tuple[int, int]) -> Tuple[int, bool]:
        """
        Processes a game action, updates the game state, and returns the reward and game status.

        This method applies the given action to the current tetromino shape, performs a hard drop,
        updates the board, calculates the reward based on cleared lines, and checks if the game is over.

        Parameters:
            action (Tuple[int, int]): The action to be performed, consisting of (position, rotation).

        Returns:
            Tuple[int, bool]: A tuple containing the calculated reward and a boolean indicating
                              whether the game is over (True) or not (False).
        """
        # Set the initial position for the shape based on the action
        pos = [action[0], 0]

        # Rotate shape 'action[1]' times
        for _ in range(action[1]):
            self.shape = rotated(self.shape)

        # Perform a hard drop of the shape to its final position
        self.shape, self.anchor = hard_drop(self.shape, pos, self.board)

        # Initialize reward and game status
        reward = 0
        done = False

        # Update the board with the new piece
        self.update_board_with_piece(True)

        # Remove completed lines and calculate reward
        cleared_lines, board = clear_lines(self.board)
        self.board = board
        self.score += cleared_lines
        reward += cleared_lines ** 2 * self.width + 1

        # Check if any blocks are in the top row, indicating game over
        if np.any(self.board[:, 0]):
            self.reset_game_state()  # Reset the game if over
            done = True
            reward -= 5
        else:
            self.initialize_new_tetromino()  # Initialize a new tetromino for the next step

        return reward, done

    def reset_game_state(self) -> np.ndarray:
        """
        Resets the game to its initial state.

        This method resets the game score to zero, reinitializes the game board to an empty state,
        selects a new tetromino piece, and returns the initial state representation used by the
        learning agent.

        Returns:
            np.ndarray: An array representing the initial state of the game, typically used as the
                        initial observation by a learning agent.
        """
        # Reset the game score to zero
        self.score = 0

        # Reinitialize the game board to an empty state
        self.board = np.zeros_like(self.board)

        # Select and initialize a new tetromino piece
        self.initialize_new_tetromino()

        # Return the initial state representation (e.g., a zero-filled array for the agent)
        return np.zeros(self.state_size, dtype=int)

    def update_board_with_piece(self, place_piece: bool) -> None:
        """
        Updates the game board by placing or removing the current tetromino piece.

        This method iterates over each block of the current tetromino shape, based on its
        anchor position, and updates the board to either place the piece (set to 1) or
        remove it (set to 0), depending on the 'place_piece' flag.

        Parameters:
            place_piece (bool): If True, places the tetromino on the board. If False, removes it.
        """
        for block_x, block_y in self.shape:
            # Calculate the board coordinates for the current block
            board_x, board_y = int(self.anchor[0] + block_x), int(self.anchor[1] + block_y)

            # Update the board if the block is within the board boundaries
            if 0 <= board_x < self.width and 0 <= board_y < self.height:
                self.board[board_x, board_y] = 1 if place_piece else 0

    def evaluate_and_update_state(self, board: np.ndarray) -> np.ndarray:
        """
        Evaluates the current state of the Tetris board and updates the game score.

        This method assesses the Tetris board by calculating the number of cleared lines,
        the total number of holes, the board's bumpiness, and the overall height. It updates
        the game's score based on the number of lines cleared and returns an array representing
        the evaluated metrics.

        Parameters:
            board (np.ndarray): The current game board.

        Returns:
            np.ndarray: An array containing the evaluated metrics: [cleared_lines, holes, bumpiness, height].
        """
        # Calculate board metrics
        cleared_lines, holes, bumpiness, height = evaluate_board_state(board)

        # Update score based on the number of cleared lines
        self.score += cleared_lines

        # Return the array of board metrics
        return np.array([cleared_lines, holes, bumpiness, height])

    def generate_all_possible_states(self) -> dict:
        """
        Generates all possible states for the current tetromino shape.

        This method explores all possible rotations and positions for the current tetromino shape
        on the board. For each valid position, it temporarily updates the board, evaluates the resulting
        state, and then resets the board to its original state. This generates a mapping of tetromino
        positions and rotations to their corresponding evaluated board states.

        Returns:
            dict: A dictionary mapping (position, rotation) tuples to evaluated board states.
        """
        # Save the current shape and anchor to restore later
        old_shape, old_anchor = self.shape, self.anchor
        states = {}

        # Explore each possible rotation of the tetromino
        for rotation in range(4):
            max_x, min_x = int(max(s[0] for s in self.shape)), int(min(s[0] for s in self.shape))

            # Explore each possible horizontal position given the current rotation
            for x in range(-min_x, self.width - max_x):
                pos = [x, 0]  # Initialize position for the shape

                # Drop the shape to the lowest possible position without collision
                while not is_occupied(self.shape, pos, self.board):
                    pos[1] += 1
                pos[1] -= 1  # Adjust position after finding the collision point

                # Temporarily update the board and evaluate the resulting state
                self.anchor = pos
                self.update_board_with_piece(True)
                states[(x, rotation)] = self.evaluate_and_update_state(self.board)
                self.update_board_with_piece(False)

                # Restore the anchor to its original position
                self.anchor = old_anchor

            # Rotate the shape for the next iteration
            self.shape = rotated(self.shape)

        # Restore the shape and anchor to their original states
        self.shape, self.anchor = old_shape, old_anchor
        return states

    def display_game_state(self, score: int) -> None:
        """
        Displays the current game state in a single window with two sections: one for the score
        and another for the Tetris board with the current tetromino piece.

        Parameters:
            score (int): The current score of the player.
        """
        # Temporarily place the current tetromino on the board for visualization
        self.update_board_with_piece(True)

        # Create a color representation of the board
        board_visual = np.array([[green if cell else black for cell in row] for row in self.board.T])

        # Remove the tetromino from the board to avoid altering the game state
        self.update_board_with_piece(False)

        # Convert the board into an image and resize for better visibility
        img_board = np.array(board_visual, dtype=np.uint8).reshape((self.height, self.width, 3))
        img_board = cv.resize(img_board, (self.width * 25, self.height * 25), interpolation=cv.INTER_NEAREST)

        # Draw grid lines on the board
        for i in range(self.height):
            img_board[i * 25, :, :] = black
        for i in range(self.width):
            img_board[:, i * 25, :] = black

        # Create an image for the score display
        img_score = np.zeros((50, self.width * 25, 3), dtype=np.uint8)
        cv.putText(img_score, f"Score: {score}", (15, 35), cv.FONT_HERSHEY_SIMPLEX, 1, white, 2)

        # Combine the score display with the game board into a single image
        img_combined = np.concatenate((img_score, img_board), axis=0)

        # Display the combined game state and score in a single window
        cv.imshow('Tetris Game', img_combined)
        cv.waitKey(1)
