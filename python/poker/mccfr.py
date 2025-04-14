import random
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import poker_core as pc

class MCCFRSolver:
    """
    Monte Carlo Counterfactual Regret Minimization (MCCFR) algorithm for solving poker games.
    Implements the chance sampling variant of MCCFR.
    """
    
    def __init__(self, game: pc.SpinGoGame):
        """
        Initialize the MCCFR solver with a SpinGoGame instance.
        
        Args:
            game: A SpinGoGame instance
        """
        self.game = game
        self.info_set_map = {}  # Maps info set string to its InfoSet object
    
    class InfoSet:
        """
        Represents an information set with cumulative regrets and strategy.
        """
        
        def __init__(self, num_actions: int):
            """
            Initialize an information set.
            
            Args:
                num_actions: Number of possible actions
            """
            self.cumulative_regrets = np.zeros(num_actions)
            self.cumulative_strategy = np.zeros(num_actions)
            self.current_strategy = np.ones(num_actions) / num_actions  # Initialize with uniform strategy
        
        def update_strategy(self):
            """
            Update the current strategy based on cumulative regrets using regret matching.
            """
            # Get positive regrets
            regrets = np.maximum(self.cumulative_regrets, 0)
            regret_sum = np.sum(regrets)
            
            # If there are positive regrets, update strategy
            if regret_sum > 0:
                self.current_strategy = regrets / regret_sum
            else:
                # If no positive regrets, use uniform strategy
                self.current_strategy = np.ones_like(regrets) / len(regrets)
        
        def get_strategy(self) -> np.ndarray:
            """
            Get the current strategy.
            
            Returns:
                A numpy array representing the strategy (probability distribution over actions)
            """
            return self.current_strategy
        
        def get_average_strategy(self) -> np.ndarray:
            """
            Get the average strategy over all iterations.
            
            Returns:
                A numpy array representing the average strategy
            """
            strategy_sum = np.sum(self.cumulative_strategy)
            if strategy_sum > 0:
                return self.cumulative_strategy / strategy_sum
            else:
                return np.ones_like(self.cumulative_strategy) / len(self.cumulative_strategy)
    
    def get_info_set_key(self, state: pc.GameState, player_idx: int) -> str:
        """
        Get a string key that uniquely identifies an information set.
        
        Args:
            state: The current game state
            player_idx: The index of the player
            
        Returns:
            A string key that uniquely identifies an information set
        """
        players = state.get_players()
        player = players[player_idx]
        hole_cards = player.get_hole_cards()
        
        # Key components
        hole_cards_str = " ".join([card.to_string() for card in hole_cards])
        community_cards = state.get_community_cards()
        community_cards_str = " ".join([card.to_string() for card in community_cards])
        
        # Include betting history (simplified)
        # In a real implementation, this would include all betting actions
        current_bets_str = " ".join([str(p.get_current_bet()) for p in players])
        
        # Game stage
        stage = int(state.get_current_stage())
        
        # Combine all information into a key
        key = f"P{player_idx}|H{hole_cards_str}|C{community_cards_str}|B{current_bets_str}|S{stage}"
        
        return key
    
    def get_info_set(self, key: str, num_actions: int) -> InfoSet:
        """
        Get the InfoSet object for a given key, creating it if it doesn't exist.
        
        Args:
            key: The information set key
            num_actions: Number of possible actions
            
        Returns:
            An InfoSet object
        """
        if key not in self.info_set_map:
            self.info_set_map[key] = self.InfoSet(num_actions)
        
        return self.info_set_map[key]
    
    def cfr(self, iterations: int, prune: bool = True):
        """
        Run MCCFR for the specified number of iterations.
        
        Args:
            iterations: Number of iterations to run
            prune: Whether to use pruning (ignore low-probability branches)
        """
        for i in range(iterations):
            if i % 100 == 0:
                print(f"Iteration {i}")
            
            # Reset the game
            self.game = pc.SpinGoGame(
                len(self.game.get_game_state().get_players()),
                self.game.get_game_state().get_players()[0].get_stack(),
                self.game.get_game_state().get_small_blind(),
                self.game.get_game_state().get_big_blind()
            )
            
            # Deal hole cards
            self.game.get_game_state().deal_hole_cards()
            
            # Run CFR for each player
            num_players = len(self.game.get_game_state().get_players())
            reach_probs = np.ones(num_players)
            
            for p in range(num_players):
                # Traverse the game tree
                self._traverse_tree(self.game.get_game_state(), p, reach_probs, prune)
            
            # Update all information set strategies
            for info_set in self.info_set_map.values():
                info_set.update_strategy()
    
    def _traverse_tree(self, state: pc.GameState, traverse_player: int, reach_probs: np.ndarray, prune: bool) -> float:
        """
        Traverse the game tree using recursive depth-first search.
        
        Args:
            state: The current game state
            traverse_player: The player for whom we're calculating CFR
            reach_probs: The probability of reaching the current state for each player
            prune: Whether to use pruning
            
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
            return self._traverse_tree(state, traverse_player, reach_probs, prune)
        elif state.get_current_stage() == pc.GameStage.FLOP and len(state.get_community_cards()) == 3:
            # Deal turn and continue
            state.deal_turn()
            return self._traverse_tree(state, traverse_player, reach_probs, prune)
        elif state.get_current_stage() == pc.GameStage.TURN and len(state.get_community_cards()) == 4:
            # Deal river and continue
            state.deal_river()
            return self._traverse_tree(state, traverse_player, reach_probs, prune)
        
        # Get legal actions
        legal_actions = state.get_legal_actions()
        num_actions = len(legal_actions)
        
        # Get info set
        info_set_key = self.get_info_set_key(state, current_player)
        info_set = self.get_info_set(info_set_key, num_actions)
        
        # Get current strategy
        strategy = info_set.get_strategy()
        
        # Initialize expected values
        action_values = np.zeros(num_actions)
        
        # If pruning is enabled, choose only one action per iteration for non-traverse player
        if prune and current_player != traverse_player:
            sampled_action = np.random.choice(num_actions, p=strategy)
            
            # Apply action and recurse
            new_state = pc.GameState(
                len(state.get_players()),
                state.get_players()[0].get_stack(),
                state.get_small_blind(),
                state.get_big_blind()
            )
            # Copy state (simplified, in practice would need deep copy)
            new_state.apply_action(legal_actions[sampled_action])
            
            # Calculate new reach probabilities
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[sampled_action]
            
            # Recurse
            action_values[sampled_action] = self._traverse_tree(new_state, traverse_player, new_reach_probs, prune)
            node_value = action_values[sampled_action]
        else:
            # For traverse player or if not pruning, consider all actions
            node_value = 0
            for a in range(num_actions):
                # Apply action and recurse
                new_state = pc.GameState(
                    len(state.get_players()),
                    state.get_players()[0].get_stack(),
                    state.get_small_blind(),
                    state.get_big_blind()
                )
                # Copy state (simplified, in practice would need deep copy)
                new_state.apply_action(legal_actions[a])
                
                # Calculate new reach probabilities
                new_reach_probs = reach_probs.copy()
                new_reach_probs[current_player] *= strategy[a]
                
                # Recurse
                action_values[a] = self._traverse_tree(new_state, traverse_player, new_reach_probs, prune)
                node_value += strategy[a] * action_values[a]
        
        # If this is the traverse player, update regrets and strategy
        if current_player == traverse_player:
            opponent_reach_prob = np.prod(reach_probs) / reach_probs[current_player]
            
            # Update regrets
            for a in range(num_actions):
                regret = opponent_reach_prob * (action_values[a] - node_value)
                info_set.cumulative_regrets[a] += regret
            
            # Update strategy
            for a in range(num_actions):
                info_set.cumulative_strategy[a] += reach_probs[current_player] * strategy[a]
        
        return node_value
    
    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        """
        Get the average strategy for all information sets.
        
        Returns:
            A dictionary mapping info set keys to average strategies
        """
        result = {}
        for key, info_set in self.info_set_map.items():
            result[key] = info_set.get_average_strategy()
        
        return result
    
    def save_strategy(self, filename: str):
        """
        Save the current strategy to a file.
        
        Args:
            filename: The name of the file to save to
        """
        strategy_dict = self.get_average_strategy()
        np.save(filename, strategy_dict)
        
    def load_strategy(self, filename: str):
        """
        Load a strategy from a file.
        
        Args:
            filename: The name of the file to load from
        """
        strategy_dict = np.load(filename, allow_pickle=True).item()
        
        # Recreate InfoSet objects from the loaded strategy
        for key, strategy in strategy_dict.items():
            info_set = self.InfoSet(len(strategy))
            info_set.cumulative_strategy = strategy * 1000  # Scale to simulate many iterations
            info_set.update_strategy()
            self.info_set_map[key] = info_set
