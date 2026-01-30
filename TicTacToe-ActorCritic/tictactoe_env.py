import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TicTacToeEnv:
    def __init__(self):
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(9,), dtype=np.int8
        )
        self.action_space = spaces.Discrete(9)
        self.reset()

    def reset(self, seed=None):
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1   # +1 or -1
        self.done = False
        return self.board.copy(), {}

    def step(self, action):
        if self.done:
            raise ValueError("Game already finished")

        if self.board[action] != 0:
            self.done = True
            return self.board.copy(), -1.0, True, False, {}

        self.board[action] = self.current_player

        if self._check_win(self.current_player):
            self.done = True
            return self.board.copy(), 1.0, True, False, {}

        if np.all(self.board != 0):
            self.done = True
            return self.board.copy(), -0.1, True, False, {}

        # switch player
        self.current_player *= -1

        return self.board.copy(), 0.0, False, False, {}

    def _check_win(self, player):
        wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        return any(all(self.board[i] == player for i in line) for line in wins)
