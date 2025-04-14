import numpy as np
import gym
from gym import spaces
import poker_core as pc
from typing import Dict, List, Tuple, Optional, Any, Union

class RLInterface:
    """
    Interface for reinforcement learning algorithms to interact with the poker game.
    Provides observation and action space conversion utilities.
    """
    
    def __init__(self, game: pc.SpinGoGame):
        """
        Initialize the RL interface with a SpinGoGame instance.
        
        Args:
            game: A SpinGoGame instance
        """
        self.game = game
        self.state = game.get_game_state()
    
    def get_observation(self, player_idx: int) -> np.ndarray:
        """
        Get an observation from the current game state for a specific player.
        
        Args:
            player_idx: The index of the player to get the observation for
            
        Returns:
            A numpy array representing the observation
        """
        # Define observation vector features
        
        # Get the game state
        state = self.game.get_game_state()
        players = state.get_players()
        community_cards = state.get_community_cards()
        num_players = len(players)
        
        # Player's hole cards (one-hot encoded)
        hole_cards_feat = np.zeros(52 * 2)  # 2 hole cards
        player = players[player_idx]
        if player.is_active():
            hole_cards = player.get_hole_cards()
            for i, card in enumerate(hole_cards):
                hole_cards_feat[i * 52 + card.get_id()] = 1
        
        # Community cards (one-hot encoded)
        comm_cards_feat = np.zeros(52 * 5)  # Up to 5 community cards
        for i, card in enumerate(community_cards):
            comm_cards_feat[i * 52 + card.get_id()] = 1
        
        # Players' stacks (normalized)
        max_stack = max([p.get_stack() for p in players])
        stacks_feat = np.zeros(num_players)
        for i, p in enumerate(players):
            stacks_feat[i] = p.get_stack() / max_stack if max_stack > 0 else 0
        
        # Players' current bets (normalized)
        max_bet = max([p.get_current_bet() for p in players])
        bets_feat = np.zeros(num_players)
        for i, p in enumerate(players):
            bets_feat[i] = p.get_current_bet() / max_bet if max_bet > 0 else 0
        
        # Players' folded status
        folded_feat = np.zeros(num_players)
        for i, p in enumerate(players):
            folded_feat[i] = 1 if p.has_folded() else 0
        
        # Game stage
        stage_feat = np.zeros(5)  # PREFLOP, FLOP, TURN, RIVER, SHOWDOWN
        stage_feat[int(state.get_current_stage())] = 1
        
        # Pot size (normalized)
        pot_feat = np.array([state.get_pot() / max_stack]) if max_stack > 0 else np.array([0])
        
        # Blind levels (normalized)
        sb = state.get_small_blind()
        bb = state.get_big_blind()
        max_blind = max(sb, bb)
        blind_feat = np.array([sb / max_blind, bb / max_blind]) if max_blind > 0 else np.array([0, 0])
        
        # Current player's position
        position_feat = np.zeros(num_players)
        position_feat[state.get_current_player_index()] = 1
        
        # Dealer position
        dealer_feat = np.zeros(num_players)
        dealer_feat[state.get_dealer_position()] = 1
        
        # Combine all features
        observation = np.concatenate([
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
        
        return observation
    
    def action_to_int(self, action: pc.Action) -> int:
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
            pc.ActionType.BET: 3,
            pc.ActionType.RAISE: 4,
            pc.ActionType.ALL_IN: 5
        }
        
        # For actions with amounts, add amount encoding
        # This is a simplified mapping - in practice, you might want to
        # discretize the amount space more carefully
        if action_type in [pc.ActionType.CALL, pc.ActionType.BET, 
                          pc.ActionType.RAISE, pc.ActionType.ALL_IN]:
            # Example: discretize into 5 amount buckets
            state = self.game.get_game_state()
            current_player = state.get_players()[state.get_current_player_index()]
            stack = current_player.get_stack()
            if stack == 0:
                return type_to_base[action_type]
            
            # Normalize amount by stack
            amount_bucket = min(int(action_amount / stack * 5), 4)
            return type_to_base[action_type] + amount_bucket
        
        return type_to_base[action_type]
    
    def int_to_action(self, action_idx: int) -> pc.Action:
        """
        Convert an integer action index to a poker Action.
        
        Args:
            action_idx: An integer action index
            
        Returns:
            A poker Action object
        """
        state = self.game.get_game_state()
        legal_actions = state.get_legal_actions()
        
        # If there's only one legal action, return it
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Map action index to type and amount
        if action_idx == 0:  # FOLD
            return pc.Action.fold()
        elif action_idx == 1:  # CHECK
            for action in legal_actions:
                if action.get_type() == pc.ActionType.CHECK:
                    return action
            # If CHECK is not legal, default to the first legal action
            return legal_actions[0]
        elif action_idx >= 2 and action_idx < 7:  # CALL with amount buckets
            for action in legal_actions:
                if action.get_type() == pc.ActionType.CALL:
                    return action
        elif action_idx >= 7 and action_idx < 12:  # BET with amount buckets
            amount_bucket = action_idx - 7
            current_player = state.get_players()[state.get_current_player_index()]
            stack = current_player.get_stack()
            
            # Calculate bet amount based on bucket and stack
            bet_amount = int(stack * (amount_bucket + 1) / 5)
            
            # Find closest legal bet
            min_diff = float('inf')
            best_action = None
            
            for action in legal_actions:
                if action.get_type() == pc.ActionType.BET:
                    diff = abs(action.get_amount() - bet_amount)
                    if diff < min_diff:
                        min_diff = diff
                        best_action = action
            
            if best_action:
                return best_action
        elif action_idx >= 12 and action_idx < 17:  # RAISE with amount buckets
            amount_bucket = action_idx - 12
            current_player = state.get_players()[state.get_current_player_index()]
            stack = current_player.get_stack()
            
            # Calculate raise amount based on bucket and stack
            raise_amount = int(stack * (amount_bucket + 1) / 5)
            
            # Find closest legal raise
            min_diff = float('inf')
            best_action = None
            
            for action in legal_actions:
                if action.get_type() == pc.ActionType.RAISE:
                    diff = abs(action.get_amount() - raise_amount)
                    if diff < min_diff:
                        min_diff = diff
                        best_action = action
            
            if best_action:
                return best_action
        elif action_idx == 17:  # ALL_IN
            for action in legal_actions:
                if action.get_type() == pc.ActionType.ALL_IN:
                    return action
        
        # Default to the first legal action if the requested action is not legal
        return legal_actions[0]
    
    def apply_action(self, action: Union[pc.Action, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply an action to the game and return the next observation, reward, done, and info.
        
        Args:
            action: A poker Action object or an integer action index
            
        Returns:
            A tuple of (observation, reward, done, info)
        """
        # Convert int action to Action if needed
        if isinstance(action, int):
            action = self.int_to_action(action)
        
        # Get current player before applying action
        state = self.game.get_game_state()
        current_player_idx = state.get_current_player_index()
        current_player = state.get_players()[current_player_idx]
        
        # Record state before action
        prev_stack = current_player.get_stack()
        prev_pot = state.get_pot()
        prev_bet = current_player.get_current_bet()
        
        # Apply the action
        state.apply_action(action)
        
        # Check if hand is over
        hand_over = state.is_hand_over()
        
        # Calculate reward (change in player's stack)
        new_player = state.get_players()[current_player_idx]
        stack_change = new_player.get_stack() - prev_stack
        
        # Add current bet contribution to the reward calculation
        pot_contribution = new_player.get_current_bet() - prev_bet
        
        # Reward is the change in stack minus pot contribution
        reward = stack_change + pot_contribution
        
        # If hand is over, check if player won
        if hand_over:
            winners = state.get_winners()
            if current_player_idx in winners:
                # Player won the hand
                if len(winners) == 1:
                    # Sole winner
                    reward = state.get_pot()
                else:
                    # Split pot
                    reward = state.get_pot() / len(winners)
        
        # Get observation for next player
        next_player_idx = state.get_current_player_index()
        observation = self.get_observation(next_player_idx)
        
        # Check if tournament is over
        tournament_over = self.game.is_tournament_over()
        
        # Info dict for additional data
        info = {
            "action_string": action.to_string(),
            "current_player": current_player_idx,
            "next_player": next_player_idx,
            "pot": state.get_pot(),
            "hand_over": hand_over,
            "tournament_over": tournament_over
        }
        
        return observation, reward, tournament_over, info


class RLEnvironment(gym.Env):
    """
    OpenAI Gym environment wrapper for the poker game.
    """
    
    def __init__(self, num_players=3, buy_in=500, small_blind=10, big_blind=20):
        """
        Initialize the environment.
        
        Args:
            num_players: Number of players in the game
            buy_in: Initial stack size for each player
            small_blind: Small blind amount
            big_blind: Big blind amount
        """
        super().__init__()
        
        # Create the game
        self.game = pc.SpinGoGame(num_players, buy_in, small_blind, big_blind)
        
        # Create the RL interface
        self.interface = RLInterface(self.game)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(18)  # 18 possible actions
        
        # Observation space dimensions
        obs_dim = (
            52 * 2 +    # Hole cards
            52 * 5 +    # Community cards
            num_players +  # Players' stacks
            num_players +  # Players' bets
            num_players +  # Players' folded status
            5 +          # Game stage
            1 +          # Pot size
            2 +          # Blind levels
            num_players +  # Current player
            num_players    # Dealer position
        )
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        
        # Current player being controlled by the RL agent
        self.current_agent_player = 0
    
    def reset(self):
        """
        Reset the environment to a new game.
        
        Returns:
            Initial observation
        """
        # Create a new game
        self.game = pc.SpinGoGame(
            len(self.game.get_game_state().get_players()),
            self.game.get_game_state().get_players()[0].get_stack(),
            self.game.get_game_state().get_small_blind(),
            self.game.get_game_state().get_big_blind()
        )
        
        # Reset the interface
        self.interface = RLInterface(self.game)
        
        # Deal hole cards
        self.game.get_game_state().deal_hole_cards()
        
        # Get observation for the current agent player
        return self.interface.get_observation(self.current_agent_player)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            A tuple of (observation, reward, done, info)
        """
        # Check if it's the agent's turn
        state = self.game.get_game_state()
        current_player_idx = state.get_current_player_index()
        
        # If it's not the agent's turn, simulate other players
        while current_player_idx != self.current_agent_player and not state.is_hand_over():
            # Simple bot policy: take the first legal action
            legal_actions = state.get_legal_actions()
            if legal_actions:
                state.apply_action(legal_actions[0])
                current_player_idx = state.get_current_player_index()
        
        # If the hand is over, deal a new hand
        if state.is_hand_over():
            self.game.play_hand()
            state = self.game.get_game_state()
            state.deal_hole_cards()
            current_player_idx = state.get_current_player_index()
            
            # If tournament is over, reset the game
            if self.game.is_tournament_over():
                return self.interface.get_observation(self.current_agent_player), 0, True, {}
        
        # Now it's the agent's turn, apply the action
        observation, reward, done, info = self.interface.apply_action(action)
        
        # Additional environment info
        info['tournament_status'] = 'ongoing' if not self.game.is_tournament_over() else 'over'
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            None
        """
        if mode == 'human':
            print(self.game.to_string())
        
        return None
