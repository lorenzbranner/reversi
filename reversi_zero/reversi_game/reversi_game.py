# std-lib import
from curses import def_shell_mode
import os
import sys
sys.path.append(os.path.dirname(__file__))

# 3-party import
import numpy as np

# projekt import
import reversi_cpp         # import reversie_cpp file 


def parse_map_file(filepath, max_players):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    player_count = int(lines[0])
    if player_count > max_players:
        raise ValueError(f"Maximal {max_players} Spieler erlaubt, aber {player_count} gefunden.")

    height, width = map(int, lines[3].split())
    raw_map_lines = lines[4:]
    
    if len(raw_map_lines) != height:
        raise ValueError(f"Erwartet {height} Zeilen, aber {len(raw_map_lines)} Zeilen in Map gefunden.")

    flat_map = []
    for line in raw_map_lines:
        tokens = line.split()
        if len(tokens) != width:
            raise ValueError(f"Erwartet {width} Spalten, aber Zeile hat {len(tokens)} Werte: {line}")
        for token in tokens:
            if token == '-':
                flat_map.append(5)
            else:
                val = int(token)
                if not (0 <= val <= player_count):
                    raise ValueError(f"Ungültiger Spielsteinwert: {val}")
                flat_map.append(val)

    board = np.array(flat_map, dtype=np.uint8).reshape((height, width))
    padded_board = np.full((15, 15), 5, dtype=np.uint8)
    padded_board[:height, :width] = board
    return padded_board, player_count


class Reversi:
    """
    
    """
    DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    use_cpp_backend: bool = True 
    
    def __init__(
        self,
        max_players: int
    ):
        self.width = 15
        self.height = 15
        self.action_size = self.width * self.height
        self.max_players = max_players
        
    def __repr__(self):
        return "Reversi"
    
    def get_initial_board(
        self, 
        filepath, 
    ):        
        return parse_map_file(filepath, self.max_players)


    def get_next_board_python(self, board, move, player):
        board = board.copy()
        x, y = move[0], move[1]
        board[y, x] = player
        for dx, dy in self.DIRECTIONS:
            path = []
            cx, cy = x + dx, y + dy
            while 0 <= cx < board.shape[1] and 0 <= cy < board.shape[0]:
                val = board[cy, cx]
                if val == 0 or val == 5:
                    break
                if val == player:
                    for px, py in path:
                        board[py, px] = player
                    break
                path.append((cx, cy))
                cx += dx
                cy += dy
        return board

    def get_next_board_cpp(self, board, move, player):
        next_board = reversi_cpp.get_next_board(board.tolist(), move, player)
        return np.array(next_board, dtype=np.uint8)

    def get_next_board(self, board, move, player):
        if self.use_cpp_backend:
            return self.get_next_board_cpp(board, move, player)
        return self.get_next_board_python(board, move, player)
    
    
    def get_valid_moves_python(self, board, player):
        valid_moves = []
        for y in range(self.height):
            for x in range(self.width):
                if self.is_valid_move(board, x, y, player):
                    valid_moves.append((x, y))
        return valid_moves
    
    def get_valid_moves_cpp(self, board, player):
        return reversi_cpp.get_valid_moves(board.tolist(), player)

    def get_valid_moves(self, board, player):
        if self.use_cpp_backend:
            return self.get_valid_moves_cpp(board=board, player=player)
        
        return self.get_valid_moves_python(board=board, player=player)
    
    
    def get_valid_moves_mask_python(
        self, 
        board, 
        player: int
    ):
        """
            Generates a binary mask indicating valid moves for the specified player.

            Each cell in the mask is set to 1 if placing a move at that position is valid
            for the given player; otherwise, it is set to 0.

            Args:
                board: The current game board.
                player (int): The player ID for whom the valid move mask is generated.

            Returns:
                np.ndarray: A 2D array of shape (height, width) with 1s at valid move positions
                and 0s elsewhere.
        """
        
        valid_moves_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                if self.is_valid_move(board, x, y, player):
                    valid_moves_mask[y, x] = 1

        return valid_moves_mask
    
    
    def get_valid_moves_mask_cpp(self, board, player: int):
        """
            Computes the valid move mask using the fast C++ backend.
            Generates a binary mask indicating valid moves for the specified player.

            Each cell in the mask is set to 1 if placing a move at that position is valid
            for the given player; otherwise, it is set to 0.

            Args:
                board: The current game board.
                player (int): The player ID for whom the valid move mask is generated.

            Returns:
                np.ndarray: A 2D array of shape (height, width) with 1s at valid move positions
                and 0s elsewhere.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        valid_moves = reversi_cpp.get_valid_moves(board.tolist(), player)
        for x, y in valid_moves:
            mask[y, x] = 1
        return mask
    
    
    def get_valid_moves_mask(self, board, player: int):
        """
           
            Generates a binary mask indicating valid moves for the specified player.
            Uses eather the cpp or the python function sed by the class variable
            
            Each cell in the mask is set to 1 if placing a move at that position is valid
            for the given player; otherwise, it is set to 0.

            Args:
                board: The current game board.
                player (int): The player ID for whom the valid move mask is generated.

            Returns:
                np.ndarray: A 2D array of shape (height, width) with 1s at valid move positions
                and 0s elsewhere.
        """
        if self.use_cpp_backend:
            return self.get_valid_moves_mask_cpp(board=board, player=player)
        
        return self.get_valid_moves_mask_python(board=board, player=player)
    
    
    def is_valid_move(self, board, x, y, player):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if board[y, x] != 0:
            return False

        for dx, dy in self.DIRECTIONS:
            if self.validate_direction(board, x, y, dx, dy, player):
                return True
        return False


    def validate_direction(self, board, x, y, dx, dy, player):
        enemy_range = set(range(1, 4)) - {player}
        x += dx
        y += dy
        found_enemy = False

        while 0 <= x < self.width and 0 <= y < self.height:
            val = board[y, x]
            if val in enemy_range:
                found_enemy = True
            elif val == player:
                return found_enemy
            else:
                return False
            x += dx
            y += dy
        return False
            
    def valid_move_player_python(self, board, player):
        for y in range(self.height):
            for x in range(self.width):
                if self.is_valid_move(board, x, y, player):
                    return True
        return False

    
    def valid_move_player(self, board, player):
        if self.use_cpp_backend:
            return reversi_cpp.valid_move_player(board.tolist(), player)
        
        return self.valid_move_player_python(board=board, player=player)


    def game_over_python(self, board, num_player) -> bool:
        for player in range(1, num_player + 1):
            if self.valid_move_player(board, player):
                return False
        return True
    
    def game_over(self, board, num_player) -> bool:
        if self.use_cpp_backend:
            return reversi_cpp.game_over(board.tolist(), num_player)
        
        return self.game_over_python(board=board, num_player=num_player)
    
    
    def get_next_player(self, player, num_players):
        if player == num_players:
            return 1
        return player + 1
    
    def get_values(self, board, num_players):
        
        player_counts = np.array([np.count_nonzero(board == p) for p in range(1, num_players + 1)])

        active_indices = np.nonzero(player_counts)[0]
        
        if len(active_indices) == 0:
            return np.zeros(self.max_players, dtype=np.float32)

        counts = player_counts[active_indices]
        sorted_indices = active_indices[np.argsort(-counts)]
        rank_points = np.array([25, 11, 5, 2])[:len(sorted_indices)]

        min_p, max_p = rank_points.min(), rank_points.max()
        if min_p == max_p:
            scaled = np.zeros_like(rank_points, dtype=np.float32)
        else:
            scaled = (rank_points - min_p) / (max_p - min_p) * 2 - 1

        result = np.zeros(num_players, dtype=np.float32)
        for idx, val in zip(sorted_indices, scaled):
            result[idx] = val

        return np.pad(result, (0, self.max_players - len(result)), mode='constant', constant_values=-1)
    
    
    def get_stone_counts(self, board, num_players):
        """
        Counts the number of stones for each player on the board.

        Parameters:
            board (np.ndarray): The game board as a 2D NumPy array.
            num_players (int): The number of currently active players in the game.

        Returns:
            np.ndarray: A 1D array of length `self.max_players` containing the number of stones 
                        for each player. Players not currently active will have a count of 0.
                        The counts are aligned to player indices 1 to `num_players` and padded
                        with zeros to reach length `self.max_players`.
        """
        player_counts = np.array([np.count_nonzero(board == p) for p in range(1, num_players + 1)])

        active_indices = np.nonzero(player_counts)[0]
        
        if len(active_indices) == 0:
            return np.zeros(self.max_players, dtype=np.float32)

        counts = player_counts[active_indices]
        
        return np.pad(counts, (0, self.max_players - len(counts)), mode='constant', constant_values=0)

    
    
    def get_encoded_board(self, board):
        """
            Converts the Reversi board into a tensor format suitable for a neural network.
            Returns an array with shape (channels, height, width).

            Channel 0: Empty fields (value 0)
            Channel 1: Player 1 stones
            Channel 2: Player 2 stones
            Channel 3: Player 3 stones (if present)
            Channel 4: Player 4 stones (if present)
            Channel 5: Blocked fields (value 5)

            If fewer than 3 or 4 players are playing, the unused player channels will be all zeros.
        """
        height, width = board.shape                                 # get board size
        channels = self.max_players + 2                             # 1 empty + 4 players + 1 blocked
        
        encoded = np.zeros((channels, height, width), dtype=np.float32)

        encoded[0] = (board == 0)                                  # Empty fields
        
        for p in range(1, self.max_players + 1):
            encoded[p] = (board == p)                              # Player p stones
        
        encoded[self.max_players + 1] = (board == 5)               # Blocked fields

        return encoded
    
    def print_board(self, board):
        print("   " + " ".join(f"{x:2}" for x in range(board.shape[1])))
        print("  +" + "---" * board.shape[1] + "+")

        for y in range(board.shape[0]):
            row = []
            for x in range(board.shape[1]):
                val = board[y, x]
                if val == 5:
                    cell = "-"
                else:
                    cell = str(val)
                row.append(f"{cell:2}")
            print(f"{y:2}| {' '.join(row)} |")

        print("  +" + "---" * board.shape[1] + "+")
