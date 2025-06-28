# std-lib import
import math
import random
import os
import time

# 3-party import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# projekt import
from reversi_game.reversi_game import Reversi
from reversi_zero import ResNet, AlphaZero
from agends.normal_mcts import MCTS


def evaluate(game, agent1, agent2, board, num_players, swap_roles=False):
    players = [agent1, agent2]
    if swap_roles:
        players.reverse()

    current_player = 1
    move_count = 0

    while not game.game_over(board, num_players):
        agent = players[(current_player - 1) % 2]

        if not game.valid_move_player(board, current_player):
            current_player = game.get_next_player(current_player, num_players)
            continue

        move = agent.get_action(board, current_player, num_players)
        move_tuple = [move % board.shape[1], move // board.shape[1]]
        board = game.get_next_board(board, move_tuple, current_player)
        current_player = game.get_next_player(current_player, num_players)
        move_count += 1

    scores = game.get_values(board, num_players)
    stone_counts = game.get_stone_counts(board, num_players)
    winner = int(np.argmax(scores)) + 1 if np.max(scores) != np.min(scores) else 0  

    return winner, scores, move_count, stone_counts



if __name__ == "__main__":
    maps_dir = "./maps/2_player_train/"
    num_games_per_map = 1

    reversi = Reversi(max_players=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(reversi, 4, 64, device, num_players=2)
    model.load_state_dict(torch.load("models/checkpoints/model_2P_10.pt", map_location=device))
    model.eval()

    agent1 = AlphaZero(model=model, game=reversi, num_searches=100)
    agent2 = MCTS(game=reversi, C = 2, num_searches=100)

    results = {1: 0, 2: 0} 

    map_files = [f for f in os.listdir(maps_dir) if f.endswith(".map")]
    print(f"[Eval] Using maps: {map_files}\n")

    for map_file in map_files:
        print(f"[Map] {map_file}")
        for i in range(num_games_per_map):
            board, num_players = reversi.get_initial_board(os.path.join(maps_dir, map_file))
            swap = (i % 2 == 1)
            winner, scores, moves, stone_counts = evaluate(reversi, agent1, agent2, board, num_players, swap_roles=swap)

            role = "AlphaZero" if (winner == 1 and not swap) or (winner == 2 and swap) else (
                   "MCTS" if winner != 0 else "Draw")
            
            player_order = "Player1: AlphaZero, Player2: MCTS" if not swap else  "Player1: MCTS, Player2: AlphaZero"
            
            print(f"  Game {i+1}/{num_games_per_map} - Moves: {moves} | {player_order} | Winner: {role} | Scores: {scores} | Stone Counts: {stone_counts}")
            results[winner] += 1
        print()

    print("==== Final Results ====")
    print(f"AlphaZero wins: {results[1]}")
    print(f"MCTS wins: {results[2]}")

