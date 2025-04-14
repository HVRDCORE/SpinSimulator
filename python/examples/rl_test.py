#!/usr/bin/env python3
"""
Simple script to test the reinforcement learning interface with our poker simulator.
This demonstrates how to use the RLEnvironment to train agents to play poker.
"""
import os
import sys
import random
import numpy as np
from pathlib import Path

# Add the parent directories to the path so we can import our module
examples_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(examples_dir)
project_dir = os.path.dirname(python_dir)
sys.path.extend([python_dir, project_dir])

# Import the poker module
try:
    import poker_core as pc
    from poker.rl_interface import RLEnvironment
except ImportError:
    print("Attempting to import the module from various locations...")
    # Try to find the module
    for path in [project_dir, python_dir, examples_dir]:
        for file in os.listdir(path):
            if file.startswith('poker_core') and file.endswith('.so'):
                print(f"Found module at {os.path.join(path, file)}")
                sys.path.insert(0, path)

    # Try importing again
    import poker_core as pc
    from poker.rl_interface import RLEnvironment


def random_agent(observation, legal_actions, player_idx):
    """
    A simple random agent that selects a random legal action.
    
    Args:
        observation: The current observation from the environment
        legal_actions: List of legal Action objects
        player_idx: The player's index
        
    Returns:
        A poker Action object
    """
    return random.choice(legal_actions)


def simple_heuristic_agent(observation, legal_actions, player_idx):
    """
    A simple heuristic agent that follows basic poker strategy.
    
    Args:
        observation: The current observation from the environment
        legal_actions: List of legal Action objects
        player_idx: The player's index
        
    Returns:
        A poker Action object
    """
    # For this simple example, it just raises or bets with 20% probability,
    # calls with 50% probability, and folds/checks the rest of the time
    action_types = [a.get_type() for a in legal_actions]
    
    # Always check if possible
    if pc.ActionType.CHECK in action_types:
        return next(a for a in legal_actions if a.get_type() == pc.ActionType.CHECK)
    
    # Bet/raise with 20% probability
    if random.random() < 0.2:
        if pc.ActionType.BET in action_types:
            return next(a for a in legal_actions if a.get_type() == pc.ActionType.BET)
        elif pc.ActionType.RAISE in action_types:
            return next(a for a in legal_actions if a.get_type() == pc.ActionType.RAISE)
    
    # Call with 50% probability
    if random.random() < 0.5 and pc.ActionType.CALL in action_types:
        return next(a for a in legal_actions if a.get_type() == pc.ActionType.CALL)
    
    # Fold if available, otherwise call
    if pc.ActionType.FOLD in action_types:
        return next(a for a in legal_actions if a.get_type() == pc.ActionType.FOLD)
    
    # Default to the first legal action
    return legal_actions[0]


def test_rl_environment(num_hands=10, seed=42):
    """
    Test the RL environment by playing hands with agents.
    
    Args:
        num_hands: Number of hands to play
        seed: Random seed
    """
    print(f"Testing RL environment with {num_hands} hands...")
    
    # Create a SpinGoGame
    game = pc.SpinGoGame(
        num_players=3,
        buy_in=500,
        small_blind=10,
        big_blind=20,
        prize_multiplier=2.0
    )
    
    # Set a seed for reproducibility
    game.set_seed(seed)
    
    # Create an RL environment
    env = RLEnvironment(game)
    
    # Dictionary to track player statistics
    stats = {
        "hands_played": 0,
        "player_wins": {0: 0, 1: 0, 2: 0},
        "total_rewards": {0: 0, 1: 0, 2: 0}
    }
    
    # Play multiple hands
    for hand_num in range(num_hands):
        print(f"\n===== Hand #{hand_num+1} =====")
        
        # Reset the game for a new hand
        game.get_game_state().reset_for_new_hand()
        
        # Deal hole cards
        game.get_game_state().deal_hole_cards()
        
        # Get the initial observation
        current_player_idx = game.get_game_state().get_current_player_index()
        observation = env.get_observation(current_player_idx)
        
        # Flags to track hand completion
        hand_over = False
        tournament_over = False
        
        # Main game loop
        while not hand_over and not tournament_over:
            # Current player state
            state = game.get_game_state()
            current_player_idx = state.get_current_player_index()
            
            # Get current player
            current_player = state.get_players()[current_player_idx]
            
            # Skip if player is not active
            if not current_player.is_active():
                print(f"Player {current_player_idx} is not active, skipping...")
                continue
            
            # Get legal actions
            legal_actions = state.get_legal_actions()
            
            # Use a heuristic agent for player 0, and random agents for others
            if current_player_idx == 0:
                action = simple_heuristic_agent(observation, legal_actions, current_player_idx)
            else:
                action = random_agent(observation, legal_actions, current_player_idx)
            
            # Apply the action
            print(f"Player {current_player_idx} performs: {action.to_string()}")
            observation, reward, done, info = env.apply_action(action)
            
            # Track rewards
            stats["total_rewards"][current_player_idx] += reward
            
            # Update flags
            hand_over = info.get("hand_over", False)
            tournament_over = info.get("tournament_over", False)
        
        # Show hand results
        stats["hands_played"] += 1
        winners = game.get_game_state().get_winners()
        
        print("\n----- Hand Results -----")
        if winners:
            for winner_idx in winners:
                stats["player_wins"][winner_idx] += 1
                print(f"Player {winner_idx} won with: {game.get_game_state().get_hand_description(winner_idx)}")
        else:
            print("No winners")
        
        print("\nCurrent Player Stats:")
        for player_idx in range(3):
            player = game.get_game_state().get_players()[player_idx]
            print(f"Player {player_idx}: Stack = {player.get_stack()}, Wins = {stats['player_wins'][player_idx]}")
        
        # Break if tournament is over
        if tournament_over:
            print("\nTournament is over!")
            winner_idx = game.get_tournament_winner()
            print(f"Tournament winner: Player {winner_idx}")
            break
    
    print("\n===== Summary =====")
    print(f"Hands played: {stats['hands_played']}")
    for player_idx in range(3):
        win_percentage = (stats["player_wins"][player_idx] / stats["hands_played"]) * 100 if stats["hands_played"] > 0 else 0
        print(f"Player {player_idx}: Wins = {stats['player_wins'][player_idx]} ({win_percentage:.2f}%), "
              f"Total Reward = {stats['total_rewards'][player_idx]}")


if __name__ == "__main__":
    test_rl_environment(num_hands=5, seed=42)