import torch
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(
        self, 
        game, 
        num_resBlocks: int = 2, 
        num_hidden: int = 100, 
        device: str = "torch" if torch.cuda.is_available else "cpu",
        num_players: int = 0
    ):
        super().__init__()
        self.device = device
        self.game_action_size = game.width * game.height                    # saves the nummber of possible moves in a game     
        self.num_players = num_players
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(num_players + 2, num_hidden, kernel_size=3, padding=1),   # the input shape will have num players + blocked + empty chanels
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.height * game.width, self.game_action_size * num_players)        # the shape will be fixed in forward
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.height * game.width, num_players),
            nn.Tanh() 
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        
        policy = self.policyHead(x)                                                 # (B, action_size * num_players)
        policy = policy.view(-1,  self.num_players, self.game_action_size)          # (B, num_players, action_size) where action size is game.width * game.height

        value = self.valueHead(x)
        
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class DummyGame:
    def __init__(self, width, height):
        self.width = width
        self.height = height


def plot_all_policy_heatmaps(policy_tensor, board_size=(15, 15)):
    """
    Erstellt einen gemeinsamen Plot mit allen Heatmaps für die Spieler.
    
    Args:
        policy_tensor (torch.Tensor | np.ndarray): Tensor der Form (1, num_players, width*height)
        board_size (tuple): Spielfeldgröße, z.B. (15, 15)
    """
    # Wenn NumPy, zu Torch konvertieren
    if isinstance(policy_tensor, np.ndarray):
        policy_tensor = torch.tensor(policy_tensor, dtype=torch.float32)

    policy_tensor = policy_tensor.squeeze(0)  # (num_players, 225)
    num_players = policy_tensor.shape[0]

    # Layout: 1 Zeile bis 3 Spieler, sonst Grid
    if num_players <= 3:
        rows, cols = 1, num_players
    else:
        cols = math.ceil(math.sqrt(num_players))
        rows = math.ceil(num_players / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < num_players:
            policy_flat = policy_tensor[i].detach().cpu().numpy()
            policy_board = policy_flat.reshape(board_size)
            im = ax.imshow(policy_board, cmap='viridis', interpolation='nearest')
            ax.set_title(f"Player {i+1}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.colorbar(im, ax=ax)
        else:
            ax.axis('off')  # Leeres Feld bei ungerader Spielerzahl

    plt.tight_layout()
    plt.show()


game = DummyGame(15, 15)
device = "cpu"
num_players = 2
model = ResNet(game, 4, 64, device, num_players)

checkpoint_path = "models/checkpoints/model_2P_10.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

board = np.array([
    [1,2,0,3,3,3,3,3,3,3,3,3,3,3,3],
    [2,1,1,3,3,3,3,3,3,3,3,3,3,3,3],
    [2,3,2,3,3,3,3,3,3,3,3,3,3,3,3],
    [0,1,2,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
])

one_hot = np.eye(4)[board]
one_hot = np.transpose(one_hot, (2, 0, 1))
one_hot = np.expand_dims(one_hot, axis=0)

encoded_board_tensor = torch.tensor(one_hot, dtype=torch.float32)

policy, value = model(encoded_board_tensor)

print(value)

plot_all_policy_heatmaps(policy)

policy_cpp_flat = np.loadtxt("weight_transfer/policy.csv", delimiter=",")
policy_cpp = policy_cpp_flat.reshape(1, 2, 225)

plot_all_policy_heatmaps(policy_cpp)