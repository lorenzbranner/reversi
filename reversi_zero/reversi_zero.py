# std-lib import
import math
import random
import os

# 3-party import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# projekt import
from reversi_game import Reversi



######## Define Models for Reversi Zero #################################################################################################################

class ResNet(nn.Module):
    def __init__(
        self, 
        game, 
        num_resBlocks: int = 1, 
        num_hidden: int = 100, 
        device: str = "torch" if torch.cuda.is_available else "cpu",
        num_players: int = 0
    ):
        super().__init__()
        self.device = device
        self.game_action_size = game.width * game.height                    # saves the nummber of possible moves in a game     
        self.num_players = num_players
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(6, num_hidden, kernel_size=3, padding=1),
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



######## Define MCTS for Rebersi Zero #################################################################################################################

class Node:
    """
        _summary_
    """
    
    def __init__(
        self, 
        game, 
        C,
        board,
        num_players: int,
        current_player: int,
        parent = None, 
        action_taken = None, 
        prior: int = 0, 
        visit_count: int = 0
    ):
        self.game = game
        self.board = board
        self.C = C
        self.num_players = num_players
        self.current_player = current_player
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        self.expandable_moves = game.get_valid_moves(board, current_player)

        self.visit_count = visit_count
        self.values = np.zeros(num_players)

    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
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
        if child.visit_count == 0:
            q_value = 0
        else:
            avg = child.values[child.current_player - 1] / child.visit_count
            q_value = 1 - ((avg + 1) / 2)
        
        return q_value + self.C * math.sqrt(self.visit_count / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for move, prob in enumerate(policy):
            if prob > 0:
                child_board = self.board.copy()
                
                move_tuple = [move // self.board.shape[1], move % self.board.shape[1]]
                child_board = self.game.get_next_board(child_board, move_tuple, self.current_player)  # TOD0
                child_player = self.game.get_next_player(self.current_player, self.num_players)
                
                child = Node(game = self.game, C=self.C, board=child_board, num_players=self.num_players, current_player= child_player, parent=self, prior=prob,  action_taken=move)
                self.children.append(child)
        return child
    
    def backpropagate(self, values):
        self.visit_count += 1
        self.values += values[:len(self.values)]
        
        if self.parent is not None:
            self.parent.backpropagate(values)  


class MCTS:
    def __init__(
        self, 
        game, 
        C, 
        model,
        dirichlet_epsiolon: float = 0.2,
        dirichlet_alpha: float = 0.2,
    ):
        self.game = game
        
        self.dirichlet_epsiolon = dirichlet_epsiolon
        self.dirichlet_alpha = dirichlet_alpha
        
        self.C = C
        self.model = model
    
    @torch.no_grad()
    def search(
        self, 
        board, 
        num_players: int,
        root_player: int, 
        num_searches: int
    ):
        
        root = Node(self.game, self.C, board, num_players, root_player)

        policy, _ = self.model(torch.tensor(self.game.get_encoded_board(board), device=self.model.device).unsqueeze(0))  # policy is shape (num_players, action_size)
        
        policy_root_player = torch.softmax(policy[0, root_player - 1], dim=0).cpu().numpy()
        policy_root_player = (1- self.dirichlet_epsiolon * policy_root_player + self.dirichlet_epsiolon * np.random.dirichlet([self.dirichlet_alpha] * self.game.action_size))
        

        valid_moves_mask = self.game.get_valid_moves_mask(board, root_player).flatten()
        
        policy_root_player *= valid_moves_mask
        policy_root_player /= np.sum(policy_root_player)        
        
        root.expand(policy_root_player)        

        for _ in range(num_searches):
            node = root
            
            while node.is_fully_expanded():     # get to the first node that is not fully expanded
                node = node.select()

            game_finished = self.game.game_over(node.board, node.num_players)       # check if in that node the game is finished

            if not game_finished:
                policys, values = self.model(
                    torch.tensor(self.game.get_encoded_board(node.board), device=self.model.device).unsqueeze(0)
                )
                
                policy = torch.softmax(policys[0, node.current_player - 1], dim=0).cpu().numpy()
                policy = (1- self.dirichlet_epsiolon * policy_root_player + self.dirichlet_epsiolon * np.random.dirichlet([self.dirichlet_alpha] * self.game.action_size))
        
                valid_moves_mask = self.game.get_valid_moves_mask(node.board, node.current_player).flatten()
                
                policy *= valid_moves_mask
                policy /= np.sum(policy)

                values = values.squeeze(0).cpu().numpy()

                node = node.expand(policy)
                
            else:
                values = self.game.get_values(node.board, node.num_players)
            
            node.backpropagate(values)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        
        return action_probs



#######   Define Reversi Zero      #############################################################################################################################

class AlphaZero:
    def __init__(
        self,
        model, 
        optimizer,
        game,
        temperature,
        board_generator,  # function or list of boards
        num_iterations: int = 3,
        num_selfPlay_iterations: int = 500,
        num_epochs: int = 4,
        batch_size: int = 64,
        C: float = 1.2, 
        dirichlet_epsiolon: float = 0.2,
        dirichlet_alpha: float = 0.2,
    ):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.temperature = temperature
        self.num_iterations = num_iterations
        self.num_selfPlay_iterations = num_selfPlay_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.board_generator = board_generator

        self.mcts = MCTS(game=game, C=C, model=model, dirichlet_alpha=dirichlet_alpha, dirichlet_epsiolon=dirichlet_epsiolon)

    def selfPlay(self, num_searches: int = 1):
        memory = []
        current_player = 1
        board, num_players = self.board_generator() if callable(self.board_generator) else random.choice(self.board_generator)

        move_count = 0
        print(f"[SelfPlay] New game started with {num_players} players.")

        while True:
            action_probs = self.mcts.search(
                board=board,
                num_players=num_players,
                root_player=current_player,
                num_searches=num_searches
            )

            memory.append((board, action_probs, current_player))

            temp_action_probs = action_probs ** (1 / self.temperature)
            move = np.random.choice(self.game.action_size, p=temp_action_probs)
            move_tuple = [move // board.shape[1], move % board.shape[1]]

            print(f"[SelfPlay] Player {current_player} chooses move {move_tuple} (flat: {move})")

            board = self.game.get_next_board(board, move_tuple, current_player)
            move_count += 1

            if self.game.game_over(board, num_players):
                print(f"[SelfPlay] Game over after {move_count} moves.")
                returnMemory = []

                final_scores = self.game.get_values(board, num_players)
                print(f"[SelfPlay] Final scores: {final_scores} scores {sum(board)}")

                for hist_board, hist_action_probs, hist_player in memory:
                    hist_outcome = final_scores[hist_player - 1]

                    returnMemory.append((
                        self.game.get_encoded_board(hist_board),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            current_player = self.game.get_next_player(current_player, num_players)


    def train(self, memory):

        random.shuffle(memory)

        for batchIdx in range(0, len(memory), self.batch_size):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx+self.batch_size)]
            board, policy_targets, value_targets = zip(*sample)

            board, policy_targets, value_targets = np.array(board), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            board = torch.tensor(board, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(board)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def learn(self):
        for iteration in range(self.num_iterations):
            memory = []

            self.model.eval()
            for SelPlay_iteration in range(self.num_selfPlay_iterations):
                memory += self.selfPlay()

            self.model.train()
            for epoch in range(self.num_epochs):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")



if __name__ == "__main__":
    max_players = 4
    
    reversi = Reversi(max_players=max_players)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNet(reversi, 4, 64, device, max_players)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    def board_generator():
        maps = ["./maps/" + f for f in os.listdir("./maps") if f.endswith(".map")]
        return reversi.get_initial_board(random.choice(maps))

    alphaZero = AlphaZero(
        model=model,
        optimizer=optimizer,
        game=reversi,
        temperature=1.25,
        num_iterations=1,
        num_selfPlay_iterations=1,
        num_epochs=500,
        batch_size=64,
        C=2,
        dirichlet_epsiolon=0.25,
        dirichlet_alpha=0.3,
        board_generator=board_generator
    )

    alphaZero.learn()
