# std-lib import
import math
import random

# 3-party import
import numpy as np

# projekt import


class Node:
    def __init__(
        self, 
        game,
        C,
        board,
        num_players,
        current_player,
        parent=None,
        move_taken=None
    ):
        self.game = game
        self.C = C
        self.board = board
        self.num_players = num_players
        self.current_player = current_player
        self.parent = parent
        self.move_taken = move_taken
        
        self.children = []
        self.expandable_moves = game.get_valid_moves(board, current_player)
        
        self.visit_count = 0
        self.values = np.zeros(num_players)
        
    def is_fully_expanded(self):
        return np.sum(len(self.expandable_moves)) == 0 and len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        q_value = 1 - ((child.values[child.current_player-1] / child.visit_count) + 1) / 2
        return q_value + self.C * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        if not self.expandable_moves:
            next_player = self.game.get_next_player(self.current_player, self.num_players)      # skip player and add a skip_child node 
            skip_child = Node(
                game=self.game,
                C=self.C,
                board=self.board.copy(),
                num_players=self.num_players,
                current_player=next_player,
                parent=self,
                move_taken=None
            )
            self.children.append(skip_child)
            return skip_child

        move = random.choice(self.expandable_moves)
        self.expandable_moves.remove(move)

        child_board = self.board.copy()
        child_board = self.game.get_next_board(child_board, move, self.current_player)
        next_player = self.game.get_next_player(self.current_player, self.num_players)

        child = Node(self.game, self.C, child_board, self.num_players, next_player, self, move)
        self.children.append(child)
        return child
    
    def simulate(self):
        rollout_board = self.board.copy()
        rollout_player = self.current_player
        
        while (True):
            if self.game.game_over(rollout_board, self.num_players):
                break

            valid_moves = self.game.get_valid_moves(rollout_board, rollout_player)

            if valid_moves:
                random_move = random.choice(valid_moves)
                rollout_board = self.game.get_next_board(rollout_board, random_move, rollout_player)

            rollout_player = self.game.get_next_player(rollout_player, self.num_players)
        
        return self.game.get_values(rollout_board, self.num_players)
            
    def backpropagate(self, values):
        self.visit_count += 1
        self.values += values
        
        if self.parent is not None:
            self.parent.backpropagate(values)  


class MCTS:
    def __init__(
        self, 
        game, 
        C,
        num_searches: int = 100
    ):
        self.game = game
        self.C = C
        self.num_searches = num_searches
        
    def search(self, board, num_players, player):
        root = Node(self.game, self.C, board, num_players, player)
        
        for search in range(self.num_searches):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            game_finished = self.game.game_over(node.board, node.num_players)
            values = None
            
            if not game_finished:
                node = node.expand()
                values = node.simulate()
            else:
                values = self.game.get_values(node.board, node.num_players)
                
            node.backpropagate(values)            
            
        action_probs = np.zeros((self.game.height, self.game.width))
        for child in root.children:
            if child.move_taken is None:
                continue
            action_probs[child.move_taken[1], child.move_taken[0]] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    
    def get_action(self, board, current_player, num_players):
        action_probs = self.search(
            board=board,
            num_players=num_players,
            player=current_player,
        )

        flat_probs = action_probs.flatten()
        move_idx = np.random.choice(len(flat_probs), p=flat_probs)

        x = move_idx % board.shape[1]
        y = move_idx // board.shape[1]
        return y * board.shape[1] + x  