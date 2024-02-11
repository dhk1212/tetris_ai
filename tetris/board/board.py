# Resources / Sources used:
# https://www.youtube.com/watch?v=os4DcbpL0Nc (last accessed 10.02.2024 / 20:48)
# https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=L8ypSXwyBds&t=2145s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=nF_crEtmpBo&t=5s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=t3fbETsIBCY (last accessed 10.02.2024 / 20:48)

from typing import Tuple

import numpy as np


def is_occupied(shape, anchor, board):
    """
    Check if a tetromino shape at a given position is colliding with the board boundaries or other pieces.

    Parameters:
    - shape (list of tuples): The tetromino shape to be checked.
    - anchor (tuple): The anchor point (x, y) for the shape on the board.
    - board (np.array): The game board represented as a 2D numpy array.

    Returns:
    - bool: True if the shape is colliding or out of bounds, False otherwise.
    """
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False


def soft_drop(shape, anchor, board):
    """
    Perform a soft drop of the tetromino shape, moving it down one position if not colliding.

    Parameters:
    - shape (list of tuples): The tetromino shape to be dropped.
    - anchor (tuple): The current anchor point of the shape.
    - board (np.array): The game board.

    Returns:
    - tuple: The shape and its new anchor point after attempting the soft drop.
    """
    new_anchor = (anchor[0], anchor[1] + 1)
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def hard_drop(shape, anchor, board):
    """
    Perform a hard drop of the tetromino shape, moving it down as far as possible without collision.

    Parameters:
    - shape (list of tuples): The tetromino shape to be dropped.
    - anchor (tuple): The current anchor point of the shape.
    - board (np.array): The game board.

    Returns:
    - tuple: The shape and its final anchor point after the hard drop.
    """
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new
        anchor = anchor_new


def clear_lines(board: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Clear completed lines from a Tetris game board and shift down any blocks above cleared lines.

    This function identifies rows in the board that are fully occupied (no zeros),
    removes them, and shifts the remaining rows down to fill the space created by the removed rows.

    Parameters:
    - board (np.ndarray): A 2D NumPy array representing the game board, where rows represent lines in the game.

    Returns:
    - Tuple[int, np.ndarray]: A tuple containing the number of lines cleared from the board and the updated game board.
    """
    if not isinstance(board, np.ndarray):
        raise ValueError("board must be a np.ndarray")

    # Identify rows that are completely filled
    complete_lines = np.all(board == 1, axis=1)  # Assuming 1 represents filled block
    lines_cleared = np.sum(complete_lines)

    # Create a new board without the completed lines
    new_board = np.delete(board, np.where(complete_lines)[0], axis=0)

    # Add empty rows on top to maintain the board size
    empty_rows = np.zeros((lines_cleared, board.shape[1]))
    new_board = np.vstack((empty_rows, new_board))

    return lines_cleared, new_board


def clear_line_dqn(board: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Clears completed lines from a Tetris board and shifts down any remaining blocks above the cleared lines.

    This function iterates through each row of the board to identify and clear rows that are completely filled.
    Each cleared row increments the count of cleared lines. The remaining rows are shifted downwards to fill
    the gaps left by the cleared lines. A new board state is generated to reflect these changes.

    The height of the board is dynamically determined from the board's shape, allowing this function to operate
    on boards of any size.

    Parameters:
    - board (np.ndarray): The Tetris game board represented as a 2D NumPy array where the shape is (width, height),
      with '1' indicating a filled block and '0' indicating an empty space.

    Returns:
    - Tuple[int, np.ndarray]: A tuple containing the number of lines that were cleared from the board, and the new
      state of the game board after clearing the lines and shifting down the blocks.
    """
    height = board.shape[1]  # Dynamically calculate the height from the board's shape
    can_clear = [np.all(board[:, i]) for i in range(height)]  # Determine rows that can be cleared
    new_board = np.zeros_like(board)  # Initialize a new board with the same shape and type
    j = height - 1  # Start from the bottom of the new board

    for i in range(height - 1, -1, -1):  # Iterate through each row from bottom to top
        if not can_clear[i]:  # If the row cannot be cleared
            new_board[:, j] = board[:, i]  # Copy it to the new board
            j -= 1  # Move up one row in the new board

    lines_cleared = sum(can_clear)  # Count the number of cleared lines

    return lines_cleared, new_board  # Return the count of cleared lines and the updated board

