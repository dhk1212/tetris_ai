import unittest

import numpy as np

from tetris.board.board import is_occupied, clear_lines, hard_drop, soft_drop


class TestBoard(unittest.TestCase):
    def test_is_occupied(self):
        shape = [(0, 0), (0, 1), (1, 0), (1, 1)]
        anchor = (0, 0)
        board = np.zeros((4, 4))
        self.assertFalse(is_occupied(shape, anchor, board))

    def test_soft_drop(self):
        shape = [(0, 0), (0, 1), (1, 0), (1, 1)]
        anchor = (0, 0)
        board = np.zeros((4, 4))
        _, new_anchor = soft_drop(shape, anchor, board)
        self.assertEqual((0, 1), new_anchor)

    def test_hard_drop(self):
        shape = [(0, 0), (0, 1), (1, 0), (1, 1)]
        anchor = (0, 0)
        board = np.zeros((4, 4))
        _, new_anchor = hard_drop(shape, anchor, board)
        self.assertEqual((0, 2), new_anchor)

    def test_clear_lines(self):
        board = np.array([
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        cleared_lines, new_board = clear_lines(board)
        self.assertEqual(2, cleared_lines)

        expected_board = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 1]
        ])
        np.testing.assert_array_equal(expected_board, new_board)
