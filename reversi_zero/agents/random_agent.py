# std-lib import

# 3-party import
import numpy as np

# projekt import


class RandomAgent:
    def __init__(self, game):
        self.game = game

    def get_action(self, board, current_player, num_players):
        valid_moves = self.game.get_valid_moves_mask(board, current_player).flatten()
        valid_indices = np.where(valid_moves > 0)[0]
        return np.random.choice(valid_indices)