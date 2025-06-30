import torch
import numpy as np
import random
from reversi_game.reversi_game import Reversi
from reversi_zero import AlphaZero, ResNet 

def dummy_board_generator(game, map_path):
    board, num_players = game.get_initial_board(map_path)
    return board, num_players, map_path

def overfit_test():
    map_path = "./maps/test_maps/overfit_test_2p.map"
    game = Reversi(max_players=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, num_resBlocks=2, num_hidden=64, device=device, num_players=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    alpha_zero = AlphaZero(
        model=model,
        optimizer=optimizer,
        game=game,
        temperature=1.0,
        num_iterations=1,
        num_selfPlay_iterations=1,
        num_epochs=100,
        batch_size=1,
        C=1.2,
        num_searches=50,
        board_generator=lambda: dummy_board_generator(game, map_path)
    )

    # Run one self-play game and take the first sample
    memory = alpha_zero.selfPlay()
    encoded_board, policy_target, value_target, player = memory[0]
    
    print("\n== Before Training ==")
    input_tensor = torch.tensor(encoded_board, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out_policy, out_value = model(input_tensor)
        softmax_policy = torch.softmax(out_policy[0, player - 1], dim=0).cpu().numpy()
        print("Initial policy:", softmax_policy.reshape(game.height, game.width))
        print("Initial value:", out_value.cpu().numpy())

    print("\n== Overfitting ==")
    alpha_zero.learn()
    
    print("\n== After Training ==")
    model.eval()
    with torch.no_grad():
        out_policy, out_value = model(input_tensor)
        softmax_policy = torch.softmax(out_policy[0, player - 1], dim=0).cpu().numpy()
        print("Trained policy:", softmax_policy.reshape(game.height, game.width))
        print("Trained value:", out_value.cpu().numpy())
        print("Target value:", np.array(value_target))
        
if __name__ == "__main__":
    overfit_test()