# Resources / Sources used:
# https://www.youtube.com/watch?v=os4DcbpL0Nc (last accessed 10.02.2024 / 20:48)
# https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/src/engine.py (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=L8ypSXwyBds&t=2145s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=nF_crEtmpBo&t=5s (last accessed 10.02.2024 / 20:48)
# https://www.youtube.com/watch?v=t3fbETsIBCY (last accessed 10.02.2024 / 20:48)

# Define tetromino shapes with their relative coordinates
shapes = {
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    'S': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}

# Define shape names for easy access and reference
shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']


def rotated(shape):
    """
    Rotate a tetromino shape 90 degrees clockwise.

    Parameters:
    - shape (list of tuples): The tetromino shape to be rotated.

    Returns:
    - list of tuples: The rotated shape.
    """
    return [(-j, i) for i, j in shape]