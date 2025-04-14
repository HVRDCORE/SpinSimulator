import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from typing import Dict, List, Tuple, Optional, Set
import time
import os
import poker_core as pc

class DeepCFRSolver:
    """
    Deep Counterfactual Regret Minimization (Deep CFR) algorithm for solving poker games.
    
    This implementation uses neural networks to approximate the advantage functions
    and strategy profiles, allowing it to handle much larger games than tabular CFR methods.
    """
    
    def __init__(self, game: pc.SpinGoGame, hidden_layers: List[int] = [256, 256],
                 learning_rate: float = 0.001, memory_size: int = 1000000):
        """
        Initialize the Deep CFR solver.
        
        Args:
            game: A SpinGoGame instance
            hidden_layers: List of hidden layer sizes for the advantage networks
            learning_rate: Learning rate for the neural networks
            memory_size: Maximum size of the reservoir buffer
        """
        self.game = game
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        # Number of players
        self.num_players = len(game.get_game_state().get_players())
        
        # Create advantage networks for each player
        self.advantage_networks = []
        for _ in range(self.num_players):
            self.advantage_networks.append(self._create_advantage_network())
            
        # Create strategy network
        self.strategy_network = self._create_strategy_network()
        
        # Create reservoir buffers for advantage memories
        self.advantage_memories = [[] for _ in range(self.num_players)]
        
        # Create reservoir buffer for strategy memory
        self.strategy_memory = []
        
        # Track iterations
        self.iterations = 0
        
    def _create_advantage_network(self) -> models.Model:
        """
        Create a neural network for the advantage function.
        
        Returns:
            A Keras Model
        """
        # Feature vector size
        # This should match the size of the feature vectors created in _state_to_features
        input_size = self._get_input_size()
        
        # Create the model
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(input_size,)))
        
        # Add hidden layers
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.2))
        
        # Output layer - one neuron per action
        model.add(layers.Dense(18, activation='linear'))  # 18 possible actions
        
        # Compile the model
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        
        return model
    
    def _create_strategy_network(self) -> models.Model:
        """
        Create a neural network for the strategy profile.
        
        Returns:
            A Keras Model
        """
        # Feature vector size
        input_size = self._get_input_size()
        
        # Create the model
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(input_size,)))
        
        # Add hidden layers
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.2))
        
        # Output layer - probability distribution over actions
        model.add(layers.Dense(18, activation='softmax'))  # 18 possible actions
        
        # Compile the model
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                     loss='categorical_crossentropy')
        
        return model
    
    def _get_input_size(self) -> int:
        """
        Get the size of the input feature vector.
        
        Returns:
            The size of the input feature vector
        """
        # Number of features in the state representation
        state = self.game.get_game_state()
        num_players = len(state.get_players())
        
        # Feature vector components:
        # - Hole cards: 52 * 2 (one-hot encoding for each card)
        # - Community cards: 52 * 5 (one-hot encoding for each card)
        # - Players' stacks: num_players
        # - Players' bets: num_players
        # - Players' folded status: num_players
        # - Game stage: 5 (one-hot encoding)
        # - Pot size: 1
        # - Blind levels: 2
        # - Current player: num_players (one-hot encoding)
        # - Dealer position: num_players (one-hot encoding)
        
        return (52 * 2) + (52 * 5) + (num_players * 5) + 8
    
    def _state_to_features(self, state: pc.GameState, player_idx: int) -> np.ndarray:
        """
        Convert a game state to a feature vector for a specific player.
        
        Args:
            state: The game state
            player_idx: The player index
            
        Returns:
            A numpy array of features
        """
        players = state.get_players()
        community_cards = state.get_community_cards()
        num_players = len(players)
        
        # Player's hole cards (one-hot encoded)
        hole_cards_feat = np.zeros(52 * 2)  # 2 hole cards
        player = players[player_idx]
        if player.is_active():
            hole_cards = player.get_hole_cards()
            for i, card in enumerate(hole_cards):
                if i < 2:  # Ensure we only process 2 cards
                    card_id = card.get_id()
                    if 0 <= card_id < 52:  # Validate card ID
                        hole_cards_feat[i * 52 + card_id] = 1
        
        # Community cards (one-hot encoded)
        comm_cards_feat = np.zeros(52 * 5)  # Up to 5 community cards
        for i, card in enumerate(community_cards):
            if i < 5:  # Ensure we only process 5 cards
                card_id = card.get_id()
                if 0 <= card_id < 52:  # Validate card ID
                    comm_cards_feat[i * 52 + card_id] = 1
        
        # Players' stacks (normalized)
        max_stack = max([p.get_stack() for p in players]) if players else 1
        stacks_feat = np.zeros(num_players)
        for i, p in enumerate(players):
            if i < num_players:  # Safety check
                stacks_feat[i] = p.get_stack() / max_stack if max_stack > 0 else 0
        
        # Players' current bets (normalized)
        max_bet = max([p.get_current_bet() for p in players]) if players else 1
        bets_feat = np.zeros(num_players)
        for i, p in enumerate(players):
            if i < num_players:  # Safety check
                bets_feat[i] = p.get_current_bet() / max_bet if max_bet > 0 else 0
        
        # Players' folded status
        folded_feat = np.zeros(num_players)
        for i, p in enumerate(players):
            if i < num_players:  # Safety check
                folded_feat[i] = 1 if p.has_folded() else 0
        
        # Game stage
        stage_feat = np.zeros(5)  # PREFLOP, FLOP, TURN, RIVER, SHOWDOWN
        stage = int(state.get_current_stage())
        if 0 <= stage < 5:  # Safety check
            stage_feat[stage] = 1
        
        # Pot size (normalized)
        pot_feat = np.array([state.get_pot() / max_stack]) if max_stack > 0 else np.array([0])
        
        # Blind levels (normalized)
        sb = state.get_small_blind()
        bb = state.get_big_blind()
        max_blind = max(sb, bb) if max(sb, bb) > 0 else 1
        blind_feat = np.array([sb / max_blind, bb / max_blind])
        
        # Current player's position
        position_feat = np.zeros(num_players)
        current_idx = state.get_current_player_index()
        if 0 <= current_idx < num_players:  # Safety check
            position_feat[current_idx] = 1
        
        # Dealer position
        dealer_feat = np.zeros(num_players)
        dealer_pos = state.get_dealer_position()
        if 0 <= dealer_pos < num_players:  # Safety check
            dealer_feat[dealer_pos] = 1
        
        # Combine all features
        features = np.concatenate([
            hole_cards_feat,
            comm_cards_feat,
            stacks_feat,
            bets_feat,
            folded_feat,
            stage_feat,
            pot_feat,
            blind_feat,
            position_feat,
            dealer_feat
        ])
        
        return features
    
    def _add_to_memory(self, memory: List, data: Tuple, t: int):
        """
        Add data to a reservoir buffer, with probability proportional to iteration.
        
        Args:
            memory: The memory buffer
            data: The data to add
            t: The current iteration
        """
        if len(memory) < self.memory_size:
            memory.append(data)
        else:
            # Reservoir sampling
            idx = random.randint(0, t)
            if idx < self.memory_size:
                memory[idx] = data
    
    def train(self, iterations: int, traversals_per_iter: int = 100, 
              advantage_training_epochs: int = 5, strategy_training_epochs: int = 5,
              batch_size: int = 128):
        """
        Train the Deep CFR solver.
        
        Args:
            iterations: Number of iterations to run
            traversals_per_iter: Number of tree traversals per iteration
            advantage_training_epochs: Number of epochs to train advantage networks per iteration
            strategy_training_epochs: Number of epochs to train strategy network per iteration
            batch_size: Batch size for neural network training
        """
        for iter_idx in range(iterations):
            start_time = time.time()
            print(f"Iteration {iter_idx+1}/{iterations}")
            
            # Track total traversals
            total_traversals = self.iterations * traversals_per_iter
            
            # Collect advantage memories for each player
            for p in range(self.num_players):
                print(f"  Collecting memories for player {p}")
                for _ in range(traversals_per_iter):
                    # Reset game
                    self._reset_game()
                    
                    # Traverse tree
                    self._traverse_tree(self.game.get_game_state(), p, total_traversals)
            
            # Train advantage networks
            for p in range(self.num_players):
                if self.advantage_memories[p]:
                    print(f"  Training advantage network for player {p}")
                    X, y = [], []
                    for state_feats, action_idx, advantage in self.advantage_memories[p]:
                        X.append(state_feats)
                        
                        # Create target vector with advantage for the specific action
                        target = np.zeros(18)  # 18 possible actions
                        target[action_idx] = advantage
                        y.append(target)
                    
                    # Convert to numpy arrays
                    X = np.array(X)
                    y = np.array(y)
                    
                    # Train the network
                    self.advantage_networks[p].fit(
                        X, y, 
                        epochs=advantage_training_epochs,
                        batch_size=batch_size,
                        verbose=0
                    )
            
            # Generate and store strategy profile
            print("  Generating strategy profile")
            for _ in range(traversals_per_iter):
                # Reset game
                self._reset_game()
                
                # Traverse tree to generate strategy
                self._traverse_tree_strategy(self.game.get_game_state(), total_traversals)
            
            # Train strategy network
            if self.strategy_memory:
                print("  Training strategy network")
                X, y = [], []
                for state_feats, strategy in self.strategy_memory:
                    X.append(state_feats)
                    y.append(strategy)
                
                # Convert to numpy arrays
                X = np.array(X)
                y = np.array(y)
                
                # Train the network
                self.strategy_network.fit(
                    X, y, 
                    epochs=strategy_training_epochs,
                    batch_size=batch_size,
                    verbose=0
                )
            
            # Update iteration counter
            self.iterations += 1
            
            # Print time taken
            time_taken = time.time() - start_time
            print(f"  Time taken: {time_taken:.2f} seconds")
    
    def _reset_game(self):
        """
        Reset the game to a new random state.
        """
        # Create a new game with the same parameters
        self.game = pc.SpinGoGame(
            len(self.game.get_game_state().get_players()),
            self.game.get_game_state().get_players()[0].get_stack(),
            self.game.get_game_state().get_small_blind(),
            self.game.get_game_state().get_big_blind()
        )
        
        # Set a random seed
        self.game.set_seed(random.randint(0, 2**31 - 1))
        
        # Deal hole cards
        self.game.get_game_state().deal_hole_cards()
    
    def _traverse_tree(self, state: pc.GameState, traverse_player: int, total_traversals: int) -> float:
        """
        Traverse the game tree to collect advantage memories.
        
        Args:
            state: The current game state
            traverse_player: The player for whom we're collecting advantages
            total_traversals: Total number of traversals performed so far
            
        Returns:
            The expected value for the traverse player
        """
        # If the hand is over, return the payoff
        if state.is_hand_over():
            winners = state.get_winners()
            if traverse_player in winners:
                # Player won (could be a split pot)
                return state.get_pot() / len(winners)
            else:
                # Player lost
                return 0
        
        # Get current player
        current_player = state.get_current_player_index()
        
        # If chance node (dealing cards), sample one outcome and continue
        if state.get_current_stage() == pc.GameStage.PREFLOP and len(state.get_community_cards()) == 0:
            # Deal flop and continue
            state.deal_flop()
            return self._traverse_tree(state, traverse_player, total_traversals)
        elif state.get_current_stage() == pc.GameStage.FLOP and len(state.get_community_cards()) == 3:
            # Deal turn and continue
            state.deal_turn()
            return self._traverse_tree(state, traverse_player, total_traversals)
        elif state.get_current_stage() == pc.GameStage.TURN and len(state.get_community_cards()) == 4:
            # Deal river and continue
            state.deal_river()
            return self._traverse_tree(state, traverse_player, total_traversals)
        
        # Get legal actions
        legal_actions = state.get_legal_actions()
        num_actions = len(legal_actions)
        
        # If no legal actions (shouldn't happen), return 0
        if num_actions == 0:
            return 0
        
        # Get state features
        state_feats = self._state_to_features(state, current_player)
        
        # If current player is the traverse player, compute advantages
        if current_player == traverse_player:
            # Use the advantage network to get action values
            action_values = self.advantage_networks[current_player].predict(
                np.array([state_feats]), verbose=0)[0]
            
            # Filter to legal actions and apply regret matching
            legal_action_values = []
            legal_action_indices = []
            
            for i, action in enumerate(legal_actions):
                # Convert action to index
                action_idx = self._action_to_index(action)
                legal_action_values.append(action_values[action_idx])
                legal_action_indices.append(action_idx)
            
            # Apply regret matching to get strategy
            strategy = self._regret_matching(np.array(legal_action_values))
            
            # Sample an action from the strategy
            action_idx = np.random.choice(len(legal_actions), p=strategy)
            action = legal_actions[action_idx]
            
            # Apply the action
            new_state = self._apply_action_and_copy_state(state, action)
            
            # Recurse
            ev = self._traverse_tree(new_state, traverse_player, total_traversals)
            
            # Compute counterfactual values and regrets
            cf_values = np.zeros(len(legal_actions))
            
            # For the chosen action, we already have the counterfactual value
            cf_values[action_idx] = ev
            
            # Compute the expected value using current strategy
            expected_value = strategy[action_idx] * ev
            
            # Store advantage for the chosen action
            advantage = cf_values[action_idx] - expected_value
            self._add_to_memory(
                self.advantage_memories[current_player],
                (state_feats, legal_action_indices[action_idx], advantage),
                total_traversals
            )
            
            return expected_value
        else:
            # For opponent nodes, use epsilon-greedy with the current approximated strategy
            # Use the advantage network to get action values
            action_values = self.advantage_networks[current_player].predict(
                np.array([state_feats]), verbose=0)[0]
            
            # Filter to legal actions and apply regret matching
            legal_action_values = []
            
            for action in legal_actions:
                # Convert action to index
                action_idx = self._action_to_index(action)
                legal_action_values.append(action_values[action_idx])
            
            # Apply regret matching to get strategy
            strategy = self._regret_matching(np.array(legal_action_values))
            
            # Sample an action from the strategy
            action_idx = np.random.choice(len(legal_actions), p=strategy)
            action = legal_actions[action_idx]
            
            # Apply the action
            new_state = self._apply_action_and_copy_state(state, action)
            
            # Recurse
            return self._traverse_tree(new_state, traverse_player, total_traversals)
    
    def _traverse_tree_strategy(self, state: pc.GameState, total_traversals: int) -> float:
        """
        Traverse the game tree to collect strategy profiles.
        
        Args:
            state: The current game state
            total_traversals: Total number of traversals performed so far
            
        Returns:
            The expected value
        """
        # If the hand is over, return 0 (we're only interested in strategies, not values)
        if state.is_hand_over():
            return 0
        
        # Get current player
        current_player = state.get_current_player_index()
        
        # If chance node (dealing cards), sample one outcome and continue
        if state.get_current_stage() == pc.GameStage.PREFLOP and len(state.get_community_cards()) == 0:
            # Deal flop and continue
            state.deal_flop()
            return self._traverse_tree_strategy(state, total_traversals)
        elif state.get_current_stage() == pc.GameStage.FLOP and len(state.get_community_cards()) == 3:
            # Deal turn and continue
            state.deal_turn()
            return self._traverse_tree_strategy(state, total_traversals)
        elif state.get_current_stage() == pc.GameStage.TURN and len(state.get_community_cards()) == 4:
            # Deal river and continue
            state.deal_river()
            return self._traverse_tree_strategy(state, total_traversals)
        
        # Get legal actions
        legal_actions = state.get_legal_actions()
        num_actions = len(legal_actions)
        
        # If no legal actions (shouldn't happen), return 0
        if num_actions == 0:
            return 0
        
        # Get state features
        state_feats = self._state_to_features(state, current_player)
        
        # Use the advantage network to get action values
        action_values = self.advantage_networks[current_player].predict(
            np.array([state_feats]), verbose=0)[0]
        
        # Create full strategy vector (for all possible actions)
        full_strategy = np.zeros(18)  # 18 possible actions
        
        # Filter to legal actions and apply regret matching
        legal_action_values = []
        legal_action_indices = []
        
        for action in legal_actions:
            # Convert action to index
            action_idx = self._action_to_index(action)
            legal_action_values.append(action_values[action_idx])
            legal_action_indices.append(action_idx)
        
        # Apply regret matching to get strategy
        strategy = self._regret_matching(np.array(legal_action_values))
        
        # Fill in the full strategy vector
        for i, action_idx in enumerate(legal_action_indices):
            full_strategy[action_idx] = strategy[i]
        
        # Store strategy profile
        self._add_to_memory(
            self.strategy_memory,
            (state_feats, full_strategy),
            total_traversals
        )
        
        # Sample an action from the strategy
        action_idx = np.random.choice(len(legal_actions), p=strategy)
        action = legal_actions[action_idx]
        
        # Apply the action
        new_state = self._apply_action_and_copy_state(state, action)
        
        # Recurse
        return self._traverse_tree_strategy(new_state, total_traversals)
    
    def _regret_matching(self, regrets: np.ndarray) -> np.ndarray:
        """
        Apply regret matching to compute a strategy.
        
        Args:
            regrets: Array of regrets for each action
            
        Returns:
            A probability distribution over actions
        """
        # Make regrets non-negative
        positive_regrets = np.maximum(regrets, 0)
        
        # Sum of positive regrets
        regret_sum = np.sum(positive_regrets)
        
        # If all regrets are negative or zero, use uniform strategy
        if regret_sum <= 0:
            return np.ones_like(regrets) / len(regrets)
        
        # Normalize by sum to get a probability distribution
        return positive_regrets / regret_sum
    
    def _action_to_index(self, action: pc.Action) -> int:
        """
        Convert a poker Action to an integer action index.
        
        Args:
            action: A poker Action object
            
        Returns:
            An integer action index
        """
        action_type = action.get_type()
        action_amount = action.get_amount()
        
        # Map action types to base indices
        type_to_base = {
            pc.ActionType.FOLD: 0,
            pc.ActionType.CHECK: 1,
            pc.ActionType.CALL: 2,
            pc.ActionType.BET: 7,
            pc.ActionType.RAISE: 12,
            pc.ActionType.ALL_IN: 17
        }
        
        # For actions with amounts, add amount encoding
        if action_type in [pc.ActionType.CALL, pc.ActionType.BET, 
                          pc.ActionType.RAISE]:
            # Get state
            state = self.game.get_game_state()
            current_player = state.get_players()[state.get_current_player_index()]
            stack = current_player.get_stack()
            
            if stack == 0:
                return type_to_base[action_type]
            
            # Normalize amount by stack
            amount_bucket = min(int(action_amount / stack * 5), 4)
            return type_to_base[action_type] + amount_bucket
        
        return type_to_base[action_type]
    
    def _apply_action_and_copy_state(self, state: pc.GameState, action: pc.Action) -> pc.GameState:
        """
        Apply an action to a copied state.
        
        Args:
            state: The original game state
            action: The action to apply
            
        Returns:
            A new game state with the action applied
        """
        # In a real implementation, we would make a deep copy of the state
        # For simplicity in this example, we'll create a new state and apply the action
        
        # Create a new state with the same basic parameters
        num_players = len(state.get_players())
        initial_stack = 0  # Will be set from player stacks
        small_blind = state.get_small_blind()
        big_blind = state.get_big_blind()
        
        new_state = pc.GameState(num_players, initial_stack, small_blind, big_blind)
        
        # Set the seed to ensure deterministic behavior
        new_state.set_seed(state.get_seed())
        
        # Apply the action to the original state
        # (this is a simplification; in a real implementation we'd maintain state integrity)
        state.apply_action(action)
        
        return state
    
    def get_strategy(self, state: pc.GameState, player_idx: int) -> np.ndarray:
        """
        Get the strategy for a given state and player.
        
        Args:
            state: The game state
            player_idx: The player index
            
        Returns:
            A probability distribution over actions
        """
        # Get state features
        state_feats = self._state_to_features(state, player_idx)
        
        # Use the strategy network to get strategy
        strategy = self.strategy_network.predict(np.array([state_feats]), verbose=0)[0]
        
        # Get legal actions
        legal_actions = state.get_legal_actions()
        
        # Filter to legal actions
        legal_action_indices = []
        for action in legal_actions:
            legal_action_indices.append(self._action_to_index(action))
        
        # Create legal strategy
        legal_strategy = np.zeros(len(legal_actions))
        for i, action_idx in enumerate(legal_action_indices):
            legal_strategy[i] = strategy[action_idx]
        
        # Normalize to get a probability distribution
        strategy_sum = np.sum(legal_strategy)
        if strategy_sum > 0:
            legal_strategy = legal_strategy / strategy_sum
        else:
            # If all probabilities are zero, use uniform strategy
            legal_strategy = np.ones_like(legal_strategy) / len(legal_strategy)
        
        return legal_strategy
    
    def save_model(self, directory: str):
        """
        Save the model to a directory.
        
        Args:
            directory: The directory to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save advantage networks
        for i, network in enumerate(self.advantage_networks):
            network.save(os.path.join(directory, f"advantage_network_{i}.h5"))
        
        # Save strategy network
        self.strategy_network.save(os.path.join(directory, "strategy_network.h5"))
    
    def load_model(self, directory: str):
        """
        Load the model from a directory.
        
        Args:
            directory: The directory to load from
        """
        # Load advantage networks
        for i in range(self.num_players):
            self.advantage_networks[i] = models.load_model(
                os.path.join(directory, f"advantage_network_{i}.h5")
            )
        
        # Load strategy network
        self.strategy_network = models.load_model(
            os.path.join(directory, "strategy_network.h5")
        )
