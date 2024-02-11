import unittest

import numpy as np

from tetris.board.state import get_bumpiness_and_height, get_holes, evaluate_board_state


class TestState(unittest.TestCase):

    def test_get_bumpiness_and_height_empty_board(self):
        board = np.zeros((4, 4), dtype=int)
        bumpiness, total_height = get_bumpiness_and_height(board)
        self.assertEqual(bumpiness, 0)
        self.assertEqual(total_height, 0)

    def test_get_bumpiness_and_height(self):
        board = np.array([
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        bumpiness, total_height = get_bumpiness_and_height(board)
        self.assertEqual(2, bumpiness)  # Adjusting based on correct logic
        self.assertEqual(14, total_height)

    def test_get_holes_no_holes(self):
        board = np.array([
            [1, 1, 1],
            [1, 1, 1]
        ])
        holes = get_holes(board)
        self.assertEqual(holes, 0)

    def test_get_holes_with_holes(self):
        board = np.array([
            [1, 0, 1],
            [1, 1, 1]
        ])
        holes = get_holes(board)
        self.assertEqual(holes, 1)

    def test_evaluate_board_state(self):
        board = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [1, 1, 1]
        ])
        state = evaluate_board_state(board)
        expected_state = np.array([2, 7, 2])
        np.testing.assert_array_equal(state, expected_state)
