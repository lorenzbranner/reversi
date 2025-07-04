# std-lib import
import math
from operator import index
import random
import os
import time

# 3-party import
import numpy as np
from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# projekt import
from reversi_game.reversi_game import Reversi


######## Define Models for Reversi Zero #################################################################################################################

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

        self.visit_count = visit_count
        self.values = np.zeros(num_players)

    def is_fully_expanded(self):
        return len(self.children) > 0       # because in MCTS for Alpha zero we will always expand all children at once 
    
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
                
                move_tuple = [move % self.board.shape[1], move // self.board.shape[1]]
                child_board = self.game.get_next_board(child_board, move_tuple, self.current_player)  # TOD0
                child_player = self.game.get_next_player(self.current_player, self.num_players)
                
                child = Node(game = self.game, C=self.C, board=child_board, num_players=self.num_players, current_player= child_player, parent=self, prior=prob,  action_taken=move)
                self.children.append(child)
    
    def backpropagate(self, values):
        self.visit_count += 1
        self.values += values[:len(self.values)]
        
        if self.parent is not None:
            self.parent.backpropagate(values)  


class MCTS:
    """
        _summary_
    """
    
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
        num_searches: int = 50
    ):
        
        root = Node(self.game, self.C, board, num_players, root_player)

        for _ in range(num_searches + 1):
            node = root
            
            while node.is_fully_expanded():     # get to the first node that is not fully expanded
                node = node.select()

            game_finished = self.game.game_over(node.board, node.num_players)       # check if in that node the game is finished

            if not game_finished:
                
                valid_moves_mask = self.game.get_valid_moves_mask(node.board, node.current_player).flatten()
                
                if np.sum(valid_moves_mask):
                
                    policys, values = self.model(
                        torch.tensor(self.game.get_encoded_board(node.board), device=self.model.device).unsqueeze(0)
                    )
                            
                    policy = torch.softmax(policys[0, node.current_player - 1], dim=0).cpu().numpy()
                    policy = (1- self.dirichlet_epsiolon) * policy + self.dirichlet_epsiolon * np.random.dirichlet([self.dirichlet_alpha] * self.game.action_size)
                    
                    policy *= valid_moves_mask
                    policy_sum = np.sum(policy)
                    
                    if policy_sum == 0:                             # fall back so we dont / 0 could happen but not very likely 
                        policy = valid_moves_mask.astype(np.float32)
                        policy /= np.sum(policy)
                    else:
                        policy /= policy_sum

                    values = values.squeeze(0).cpu().numpy()
                    node.expand(policy)
                else:
                    child_player = self.game.get_next_player(node.current_player, node.num_players)                                                                     # we need to skip this player because he has no valid moves 
                    child = Node(game=self.game, C=self.C, board=node.board.copy(), num_players=node.num_players, current_player=child_player, parent=node)             # create child node with the same board but next player
                    
                    _, values = self.model(
                        torch.tensor(self.game.get_encoded_board(node.board), device=self.model.device).unsqueeze(0)
                    )
                    
                    node.children.append(child)
                    values = values.squeeze(0).cpu().numpy()
                    child.backpropagate(values)
            else:
                values = self.game.get_values(node.board, node.num_players)
            
            node.backpropagate(values)
        
        action_probs = np.zeros(self.game.action_size)
        
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count    
      
        action_probs /= np.sum(action_probs)
        
        return action_probs

    @torch.no_grad()
    def search_parallel(
        self, 
        spGames,
        num_searches: int = 50
    ):
        
        for spg in spGames:
            spg.root = Node(self.game, self.C, spg.board, spg.num_players, spg.current_player)

        for search in range(num_searches):
            for spg in spGames:  # go trought all of the games 
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                game_finished = self.game.game_over(node.board, node.num_players)       # check if in that node the game is finished

                if not game_finished:
                    spg.node = node
                else:
                    values = self.game.get_values(node.board, node.num_players)
                    node.backpropagate(values)
               
            games_still_running = [index for index in range(len(spGames)) if spGames[index].node is not None]    # note that if we terminate the spg.node will always stay None

            if len(games_still_running) > 0:
                encoded_boards = np.stack([self.game.get_encoded_board(spGames[inndex].node.board) for inndex in games_still_running])

                policy, values = self.model(
                    torch.tensor(encoded_boards, device=self.model.device)
                )
                

                for i, index in enumerate(games_still_running):
                    node = spGames[index].node
                    valid_moves_mask = self.game.get_valid_moves_mask(node.board, node.current_player).flatten()
                    spg_policy, spg_values = policy[i], values[i]
                
                    spg_policy = torch.softmax(spg_policy[0, spg.current_player - 1], dim=0).cpu().numpy()
                    spg_policy = (1- self.dirichlet_epsiolon) * spg_policy + self.dirichlet_epsiolon * np.random.dirichlet([self.dirichlet_alpha] * self.game.action_size)
                    
                    if np.sum(valid_moves_mask): 
                    
                        spg_policy *= valid_moves_mask
                        policy_sum = np.sum(spg_policy)
                        
                        if policy_sum == 0:                                         # fall back so we dont / 0 could happen but not very likely 
                            spg_policy = valid_moves_mask.astype(np.float32)
                            spg_policy /= np.sum(spg_policy)
                        else:
                            spg_policy /= policy_sum

                        node.expand(spg_policy)
                        spg_values = spg_values.squeeze(0).cpu().numpy()
                        node.backpropagate(spg_values)
                    else:
                        child_player = self.game.get_next_player(node.current_player, node.num_players)                                                                     # we need to skip this player because he has no valid moves 
                        child = Node(game=self.game, C=self.C, board=node.board.copy(), num_players=node.num_players, current_player=child_player, parent=node)             # create child node with the same board but next player and 
                    
                        node.children.append(child)
                        spg_values = spg_values.squeeze(0).cpu().numpy()
                        child.backpropagate(spg_values)
        
        # here we dont return something we will handel this in the self play 

#######   Define Reversi Zero      #############################################################################################################################
def log(tag, message, tag_width=7, indent_after_tag=24):
    tag_str = f"[{tag:<{tag_width}}]"
    spacing = " " * (indent_after_tag - len(tag_str))
    print(f"{tag_str}{spacing}{message}")

class SelfPlayGame:
    """
    Container class for a single self-play instance in AlphaZero training.

    Attributes:
        board (np.ndarray): The current board state of the game.
        current_player (int): The index (1-based) of the player whose turn it is.
        num_players (int): Total number of players in the game.
        root (Node or None): The root node of the MCTS tree for this game.
        node (Node or None): The current node being expanded during MCTS.
    """
    def __init__(
        self, 
        board, 
        current_player, 
        num_players
    ):
        self.board = board
        self.current_player = current_player
        self.num_players = num_players
        self.memory = []
        self.root = None
        self.node = None
        

class AlphaZero:
    """
        AlphaZero training engine for multi-player games using self-play and MCTS.

        This class encapsulates the full training pipeline for AlphaZero, including self-play game generation,
        model training, checkpointing, and inference. It supports both sequential and parallel self-play modes
        and works with arbitrary board games as long as the provided `game` class exposes the necessary interface.

        Attributes:
            model (nn.Module): Neural network with a policy and value head.
            game (object): Game logic handler with move validation, transition, and scoring functions.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            temperature (float): Exploration temperature for early-game moves.
            board_generator (callable or list): Function or list yielding (board, num_players, map_name) tuples.
            num_iterations (int): Number of full training iterations.
            num_selfPlay_iterations (int): Number of self-play games per training iteration.
            num_epochs (int): Number of epochs per training iteration.
            batch_size (int): Batch size for training.
            C (float): MCTS exploration constant.
            num_searches (int): Number of simulations per MCTS decision.
            dirichlet_epsiolon (float): Noise ratio for MCTS root exploration.
            dirichlet_alpha (float): Dirichlet distribution alpha parameter.
            greedy_move_count (int): Number of moves before switching to greedy temperature.
            use_parralel_self_play (bool): Whether to run multiple games in parallel during self-play.
            num_parallel_games (int): Number of parallel games if `use_parralel_self_play` is True.
    """
    
    def __init__(
        self,
        model, 
        game,
        optimizer = None,
        temperature: float = 1,
        board_generator: callable = None,  # function or list of boards
        num_iterations: int = 3,
        num_selfPlay_iterations: int = 500,
        num_epochs: int = 4,
        batch_size: int = 64,
        C: float = 1.2, 
        num_searches: int = 50,
        dirichlet_epsiolon: float = 0.2,
        dirichlet_alpha: float = 0.2,
        greedy_move_count: int = 20,
        use_parralel_self_play: bool = True,
        num_parallel_games : int = 5
    ):
        """
            Initializes the AlphaZero training loop with model, MCTS, and training parameters.

            Args:
                model (nn.Module): A neural network with a policy and value head.
                game (object): Game environment providing rules, board encoding, and transitions.
                optimizer (torch.optim.Optimizer, optional): Optimizer used to train the model.
                temperature (float): Initial temperature for exploration in early moves.
                board_generator (callable or list): Either a function that returns (board, num_players, map_name) 
                                                    or a list of such tuples.
                num_iterations (int): Number of training iterations (each consisting of self-play + training).
                num_selfPlay_iterations (int): Number of self-play games per training iteration.
                num_epochs (int): Number of epochs per training iteration.
                batch_size (int): Batch size for model training.
                C (float): Exploration constant used in MCTS.
                num_searches (int): Number of MCTS simulations per move.
                dirichlet_epsiolon (float): Strength of Dirichlet noise added to policy priors.
                dirichlet_alpha (float): Alpha parameter for Dirichlet noise.
                greedy_move_count (int): Number of initial moves before switching to lower temperature.
                use_parralel_self_play (bool): If True, run multiple games in parallel during self-play.
                num_parallel_games (int): Number of parallel games to run during self-play if enabled.
        """
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.temperature = temperature
        self.num_iterations = num_iterations
        self.num_selfPlay_iterations = num_selfPlay_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.board_generator = board_generator
        self.num_searches = num_searches

        self.greedy_move_count = greedy_move_count

        self.use_parralel_self_play = use_parralel_self_play
        self.num_parallel_games = num_parallel_games

        self.mcts = MCTS(game=game, C=C, model=model, dirichlet_alpha=dirichlet_alpha, dirichlet_epsiolon=dirichlet_epsiolon)


    def self_play_normal(self):
        """
            Plays a single full self-play game using MCTS and returns the training data.

            This method initializes a new game (board, number of players, and map) and simulates 
            it until the end. In each turn, it uses Monte Carlo Tree Search (MCTS) to estimate 
            action probabilities, selects a move based on a temperature-controlled distribution, 
            and applies the move to the board. All game steps are stored along with the final 
            outcome for supervised learning.

            Returns:
                List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]: A list of training samples, 
                each containing:
                    - encoded board state (as input to the neural network),
                    - policy target (MCTS action distribution),
                    - value target (final outcome vector for all players),
                    - player index (1-based) who made the decision.
        """
        memory = []
        current_player = 1
        board, num_players, map_name = self.board_generator() if callable(self.board_generator) else random.choice(self.board_generator)

        move_count = 0
        log("SelfPlay", f"New game started with {num_players} players on map: {os.path.basename(map_name)}")

        while True:            
            if not self.game.valid_move_player(board, current_player):  # if the current player has no valid move skip him and go to the next player 
               current_player = self.game.get_next_player(current_player, num_players)
               continue

            action_probs = self.mcts.search(
                board=board,
                num_players=num_players,
                root_player=current_player,
                num_searches=self.num_searches
            )

            memory.append((board, action_probs, current_player))
            
            if move_count < self.greedy_move_count:     # after  self.greedy_move_count we will scale the temperature down to get 
                temperature = self.temperature 
            else:
                temperature = 0.1
            
            temp_action_probs = action_probs ** (1 / temperature)
            move = np.random.choice(self.game.action_size, p=temp_action_probs /  np.sum(temp_action_probs))
            move_tuple = [move % (board.shape[1] ), move // (board.shape[1])]

            board = self.game.get_next_board(board, move_tuple, current_player)
            move_count += 1

            if self.game.game_over(board, num_players):
                log("SelfPlay", f"Game over after {move_count} moves.")

                returnMemory = []

                final_scores = self.game.get_values(board, num_players)
                log("SelfPlay", f"Final scores: {final_scores}.")

                for hist_board, hist_action_probs, hist_player in memory:

                    returnMemory.append((
                        self.game.get_encoded_board(hist_board),
                        hist_action_probs,
                        final_scores,
                        hist_player
                    ))
                return returnMemory

            current_player = self.game.get_next_player(current_player, num_players)


    def parallel_self_play(self):
        """
            Executes multiple self-play games in parallel using MCTS.

            This method initializes a list of `SelfPlayGame` instances, each representing 
            a separate game with potentially different maps and player counts. In each turn, 
            all active games advance one move using MCTS-guided decisions. The results of all 
            games (state, policy, outcome) are collected into a single replay buffer for training.

            Returns:
                List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]: A list of tuples,
                each containing:
                    - the encoded board state,
                    - the target policy as a probability distribution over actions,
                    - the final game result (value) for each player,
                    - the current player index (1-based).
        """
        return_memory = []
        self_play_games = []
        current_player = 1                              # in all of the games we will start with player 1 :D
        
        for i in range(self.num_parallel_games):        # here we fill our list of self play games we play in parrallel, all games will be in SelfPlayGame class to save all the current params
            board, num_players, map_name = self.board_generator() if callable(self.board_generator) else random.choice(self.board_generator)
            self_play_games += [SelfPlayGame(board=board, num_players=num_players, current_player= current_player)]
            
        move_count = 0
        log("SelfPlay", f"{self.num_parallel_games} new games started in parralel")

        while len(self_play_games) > 0:
            # we need to make sure that in all of our games the current palyer has a valid move
            for spg in self_play_games:
                while not self.game.valid_move_player(spg.board, spg.current_player):   # we dont need to see if the game is over because at the beggining of all iterations the game is not over because we checked it at the end 
                    spg.current_player = self.game.get_next_player(spg.current_player, spg.num_players)
            
            self.mcts.search_parallel(
                spGames=self_play_games,
                num_searches= self.num_searches
            )
            
            for i in range(len(self_play_games))[::-1]:
                spg = self_play_games[i]

                action_probs = np.zeros(self.game.action_size)
                
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                
                action_probs /= np.sum(action_probs)
                spg.memory.append((spg.root.board.copy(), action_probs.copy(), spg.current_player))
                
                if move_count < self.greedy_move_count:         # after  self.greedy_move_count we will scale the temperature down 
                    temperature = self.temperature 
                else:
                    temperature = 0.1

                temp_action_probs = action_probs ** (1 / temperature)
                
                move = np.random.choice(self.game.action_size, p=temp_action_probs /  np.sum(temp_action_probs))
                move_tuple = [move % (spg.board.shape[1] ), move // (spg.board.shape[1])]

                spg.board = self.game.get_next_board(spg.board, move_tuple, spg.current_player)

                if self.game.game_over(spg.board, spg.num_players):
                    for hist_board, hist_action_probs, hist_player in spg.memory:
                        final_scores = self.game.get_values(spg.board, spg.num_players)
                         
                        return_memory.append((
                            self.game.get_encoded_board(hist_board),
                            hist_action_probs,
                            final_scores,
                            hist_player
                        ))
                    del self_play_games[i]

            for spg in self_play_games:
                spg.current_player = self.game.get_next_player(spg.current_player, spg.num_players)
            move_count += 1

        return return_memory
            
            
    def self_play(self):
        """
            Executes a single self-play iteration using the configured strategy.

            Depending on the configuration (`self.use_parralel_self_play`), this method either runs:
            - `parallel_self_play()` for executing multiple games simultaneously using batched MCTS evaluations, or
            - `self_play_normal()` for running a single game sequentially.

            Returns:
                List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]: A list of training samples from the self-play game(s), 
                each containing:
                    - encoded board state,
                    - target action probability distribution (policy),
                    - final game result (value for each player),
                    - current player index.
        """            
        if self.use_parralel_self_play:
            return self.parallel_self_play()
        return self.self_play_normal()
        
        
    def train(self, memory):
        """
        Trains the model using self-play experience data.

        This function converts the collected memory (game states, policy targets, value targets, 
        and current players) into a PyTorch DataLoader and trains the neural network over multiple 
        mini-batches using KL divergence for the policy head and mean squared error for the value head.

        Args:
            memory (List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]): 
                A list of tuples containing:
                - encoded board state as NumPy array,
                - target policy as a probability distribution,
                - final game result (value target),
                - current player index.

        Returns:
            Tuple[float, float]: 
                The average policy loss and average value loss over all batches.
        """
        if len(memory) == 0:
            return 0.0, 0.0

        board, policy_targets, value_targets, current_player = zip(*memory)

        board_tensor = torch.tensor(np.array(board), dtype=torch.float32, device=self.model.device)
        policy_tensor = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
        value_tensor = torch.tensor(np.array(value_targets), dtype=torch.float32, device=self.model.device)
        player_tensor = torch.tensor(np.array(current_player), dtype=torch.long, device=self.model.device) - 1  # 0-based

        dataset = TensorDataset(board_tensor, policy_tensor, value_tensor, player_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0

        for board_batch, policy_batch, value_batch, player_idx_batch in loader:
            out_policy, out_value = self.model(board_batch)
            selected_policy = out_policy[torch.arange(out_policy.size(0)), player_idx_batch]

            log_probs = F.log_softmax(selected_policy, dim=1)
            policy_loss = F.kl_div(log_probs, policy_batch, reduction="batchmean")
            value_loss = F.mse_loss(out_value, value_batch.reshape(out_value.shape))
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_batches += 1

        avg_policy_loss = total_policy_loss / total_batches
        avg_value_loss = total_value_loss / total_batches

        return avg_policy_loss, avg_value_loss
        
        
    def learn(
        self,
        checkpoint_folder: str = "models/checkpoints/",
        checkpoint_start: int = 0,
        checkpoint_iteration: int = 10,
        train_log_iteration: int = 25,
    ):
        """
            Runs the full AlphaZero training loop over multiple iterations.

            In each iteration, this method:
            1. Generates self-play data via multiple games using MCTS.
            2. Trains the neural network using the collected data.
            3. Saves model and optimizer checkpoints at specified intervals.

            Args:
                checkpoint_folder (str): Directory where model checkpoints will be saved.
                checkpoint_start (int): Start iteration (useful for resuming training).
                checkpoint_iteration (int): Interval for saving model/optimizer checkpoints.
                train_log_iteration (int): Interval for logging training loss during epochs.

            Side effects:
                - Saves model and optimizer weights to `checkpoint_folder`.
                - Logs training progress and performance via the `log` function.
        """
        for iteration in range(checkpoint_start + 1, self.num_iterations + 1):
            
            print(f"\n===== [Iteration {iteration}/{self.num_iterations}] =====")
            
            memory = []

            start_time = time.time()
            self.model.eval()
            for SelPlay_iteration in range(self.num_selfPlay_iterations):
                memory += self.self_play()

            duration = time.time() - start_time
            log("SelfPlay", f"Completed {self.num_selfPlay_iterations if not self.use_parralel_self_play else self.self.num_selfPlay_iterations * self.num_parallel_games} games in {duration:.2f}s")
            log("SelfPlay", f"Total samples collected: {len(memory)}")
          
            print() # one new line
          
            log("Train", f"Start")
            self.model.train()
            for epoch in range(1, self.num_epochs + 1):
                avg_policy_loss, avg_value_loss = self.train(memory)
                
                if not epoch % train_log_iteration:
                    log("Train", f"Epoch {epoch}/{self.num_epochs}")
                    log("Train", f"Avg Policy Loss: {avg_policy_loss:.4f} | Avg Value Loss: {avg_value_loss:.4f}")

            if not iteration % checkpoint_iteration:
                os.makedirs(checkpoint_folder, exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_folder + f"model_{self.game.max_players}P_{iteration}.pt")
                torch.save(self.optimizer.state_dict(),  checkpoint_folder + f"optimizer_{self.game.max_players}P_{iteration}.pt")

                log("Checkpoint", f"Model and optimizer saved at iteration {iteration}")

        os.makedirs(checkpoint_folder, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_folder + f"model_{self.game.max_players}P_final.pt")
        torch.save(self.optimizer.state_dict(),  checkpoint_folder + f"optimizer_{self.game.max_players}P_final.pt")


    @torch.no_grad()
    def get_action(
        self, 
        board, 
        current_player, 
        num_players
    ):  
        """
            Selects the best action for the current player using MCTS.

            This function performs a forward pass of MCTS from the given board state, 
            masks out invalid moves, and returns the move with the highest probability.

            Args:
                board (np.ndarray): Current game board state.
                current_player (int): Index of the player taking the move.
                num_players (int): Total number of players in the game.

            Returns:
                int: The selected move index (flattened coordinate).
        """
        with torch.no_grad():
            action_probs = self.mcts.search(
                board=board,
                num_players=num_players,
                root_player=current_player,
                num_searches=self.num_searches
            )
            valid_moves_mask = self.game.get_valid_moves_mask(board, current_player).flatten()
            action_probs *= valid_moves_mask
            action_probs /= np.sum(action_probs) 

            move = np.argmax(action_probs)
            return move
    
    

#######   MAIN WHAT DOSE MAIN MEEEEEEAAN ATTT ALL ?!?!?   #############################################################################################################################################################################################################################################################
    
    
if __name__ == "__main__":
    
    max_players = 2
    maps_path = "./maps/2_player_train/"
    from_checkpoint = True
    checkpoint = 20
    
    reversi = Reversi(max_players=max_players)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not from_checkpoint:
        model = ResNet(reversi, 4, 64, device, max_players)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    else:
        model = ResNet(reversi, 4, 64, device, max_players)
        model.load_state_dict(torch.load(f"models/checkpoints/model_2P_{checkpoint}.pt", map_location=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        optimizer.load_state_dict(torch.load(f"models/checkpoints/optimizer_2P_{checkpoint}.pt", map_location=device))

    def board_generator():
        maps = [maps_path + f for f in os.listdir(maps_path) if f.endswith(".map")]
        selected_map = random.choice(maps)
        board, num_players = reversi.get_initial_board(selected_map)
        return board, num_players, selected_map
    
    alphaZero = AlphaZero(
        model=model,
        optimizer=optimizer,
        game=reversi,
        temperature=1.25,
        num_iterations=200,
        num_selfPlay_iterations=80,
        num_searches = 400,
        num_epochs=300,
        batch_size=256,
        C=2,
        dirichlet_epsiolon=0.25,
        dirichlet_alpha=0.3,
        board_generator=board_generator
    )

    alphaZero.learn(
        checkpoint_start = 0 if not from_checkpoint else checkpoint
    )