# Resources / Sources used:
# https://www.youtube.com/watch?v=os4DcbpL0Nc (last accessed 10.02.2024 / 20:48)
# https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=L8ypSXwyBds&t=2145s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=nF_crEtmpBo&t=5s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=t3fbETsIBCY (last accessed 10.02.2024 / 20:48)

import numpy as np

from .board import clear_line_dqn


def get_bumpiness_and_height(board):
    """
    Calculate the bumpiness and total height of the Tetris board.

    Bumpiness is defined as the sum of the absolute differences in heights between adjacent columns.
    Total height is the sum of the heights of all columns.

    Parameters:
    - board (np.array): The game board.

    Returns:
    - tuple: (bumpiness, total_height)
    """
    total_height = 0
    bumpiness = 0

    column_heights = []

    for col in range(board.shape[1]):
        column = board[:, col]
        filled_cells = np.where(column > 0)[0]
        height = 0 if len(filled_cells) == 0 else board.shape[0] - np.min(filled_cells)
        column_heights.append(height)
        total_height += height

    for i in range(len(column_heights) - 1):
        bumpiness += abs(column_heights[i] - column_heights[i + 1])

    return bumpiness, total_height


def get_holes(board):
    """
    Count the number of holes on the Tetris board.

    A hole is defined as an empty space such that there is at least one block above it.

    Parameters:
    - board (np.array): The game board.

    Returns:
    - int: The number of holes on the board.
    """
    inverted_board = np.flipud(board)
    holes = 0
    for col in range(board.shape[1]):
        column = inverted_board[:, col]
        holes += np.sum(np.cumsum(column) > 0) - np.sum(column)
    return holes


def evaluate_board_state(board):
    """
    Evaluate the current board of the Tetris board, returning metrics useful for AI.

    Parameters:
    - board (np.array): The game board.

    Returns:
    - np.array: An array containing evaluated metrics [bumpiness, total_height, holes].
    """
    cleared_lines, board = clear_line_dqn(board)
    bumpiness, total_height = get_bumpiness_and_height(board)
    holes = get_holes(board)
    return np.array([cleared_lines, bumpiness, total_height, holes])
