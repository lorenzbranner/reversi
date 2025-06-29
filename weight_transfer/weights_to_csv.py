import torch
import os
import numpy as np
import csv

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

game = DummyGame(15, 15)
device = "cpu"
num_players = 2
model = ResNet(game, 4, 64, device, num_players)

checkpoint_path = "models/checkpoints/model_2P_10.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

os.makedirs("weight_transfer/weights_csv", exist_ok=True)

for name, param in model.named_parameters():
    if param.requires_grad:
        file_name = f"weight_transfer/weights_csv/{name}.csv"
        print(f"Speichere: {file_name}")
        data = param.detach().cpu().numpy()
        
        with open(file_name, mode='w', newline='') as f:
            writer = csv.writer(f)
            if data.ndim == 1:
                writer.writerow(data)
            else:
                writer.writerows(data.reshape(data.shape[0], -1))