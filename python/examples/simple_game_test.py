#!/usr/bin/env python3
"""
Simple script to test basic poker game functionality using the poker_core module.
This demonstrates how to use the core simulator without the TensorFlow-dependent RL code.
"""
import os
import sys
import random
import time
from pathlib import Path

# Add the parent directories to the path so we can import our module
examples_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(examples_dir)
project_dir = os.path.dirname(python_dir)
sys.path.extend([python_dir, project_dir])

# Import the poker module
try:
    import poker_core as pc
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


def random_agent(legal_actions, player_idx):
    """
    A simple random agent that selects a random legal action.
    
    Args:
        legal_actions: List of legal Action objects
        player_idx: The player's index
        
    Returns:
        A poker Action object
    """
    return random.choice(legal_actions)


def simple_heuristic_agent(legal_actions, player_idx, state):
    """
    A simple heuristic agent that follows basic poker strategy.
    
    Args:
        legal_actions: List of legal Action objects
        player_idx: The player's index
        state: The current game state
        
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


def print_player_info(state, player_idx):
    """Print player information"""
    player = state.get_players()[player_idx]
    hole_cards = player.get_hole_cards()
    hole_cards_str = ", ".join([card.to_string() for card in hole_cards])
    
    print(f"Player {player_idx}: Stack={player.get_stack()}, "
          f"Cards={hole_cards_str}, "
          f"Folded={player.has_folded()}, "
          f"All-in={player.is_all_in()}")


def test_poker_game(num_hands=5, seed=42):
    """
    Test the poker game by playing hands between agents.
    
    Args:
        num_hands: Number of hands to play
        seed: Random seed
    """
    print(f"Testing poker game with {num_hands} hands...")
    
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
    
    # Dictionary to track player statistics
    stats = {
        "hands_played": 0,
        "player_wins": {0: 0, 1: 0, 2: 0},
        "hands_folded": {0: 0, 1: 0, 2: 0}
    }
    
    # Play multiple hands
    for hand_num in range(num_hands):
        print(f"\n===== Hand #{hand_num+1} =====")
        
        # Reset the game for a new hand
        state = game.get_game_state()
        state.reset_for_new_hand()
        
        # Deal hole cards
        state.deal_hole_cards()
        
        # Show initial state
        print("Initial state:")
        for i in range(3):
            print_player_info(state, i)
        
        print(f"Dealer position: {state.get_dealer_position()}")
        print(f"Small blind: {state.get_small_blind()}, Big blind: {state.get_big_blind()}")
        
        # Play the hand
        hand_over = False
        while not hand_over:
            # Get current player and legal actions
            current_player_idx = state.get_current_player_index()
            current_player = state.get_players()[current_player_idx]
            legal_actions = state.get_legal_actions()
            
            # Skip if player is not active
            if not current_player.is_active():
                continue
            
            # Current stage and pot
            stage = state.get_current_stage()
            pot = state.get_pot()
            community_cards = state.get_community_cards()
            community_str = ", ".join([card.to_string() for card in community_cards])
            
            print(f"\nStage: {stage}, Pot: {pot}, Community cards: {community_str}")
            print(f"Player {current_player_idx}'s turn, Stack: {current_player.get_stack()}")
            
            # Use a heuristic agent for player 0, and random agents for others
            if current_player_idx == 0:
                action = simple_heuristic_agent(legal_actions, current_player_idx, state)
            else:
                action = random_agent(legal_actions, current_player_idx)
            
            # Apply the action
            print(f"Player {current_player_idx} performs: {action.to_string()}")
            state.apply_action(action)
            
            # Track fold statistics
            if action.get_type() == pc.ActionType.FOLD:
                stats["hands_folded"][current_player_idx] += 1
            
            # Check if hand is over
            hand_over = state.is_hand_over()
        
        # Show hand results
        stats["hands_played"] += 1
        winners = state.get_winners()
        
        print("\n----- Hand Results -----")
        if winners:
            for winner_idx in winners:
                stats["player_wins"][winner_idx] += 1
                player = state.get_players()[winner_idx]
                print(f"Player {winner_idx} won with: {state.get_hand_description(winner_idx)}")
                print(f"Final stack: {player.get_stack()}")
        else:
            print("No winners")
        
        print("\nCurrent Player Stats:")
        for player_idx in range(3):
            player = state.get_players()[player_idx]
            print(f"Player {player_idx}: Stack = {player.get_stack()}, "
                  f"Wins = {stats['player_wins'][player_idx]}, "
                  f"Folds = {stats['hands_folded'][player_idx]}")
        
        # Check if tournament is over
        if game.is_tournament_over():
            print("\nTournament is over!")
            winner_idx = game.get_tournament_winner()
            print(f"Tournament winner: Player {winner_idx}")
            break
    
    print("\n===== Summary =====")
    print(f"Hands played: {stats['hands_played']}")
    for player_idx in range(3):
        win_percentage = (stats["player_wins"][player_idx] / stats["hands_played"]) * 100 if stats["hands_played"] > 0 else 0
        fold_percentage = (stats["hands_folded"][player_idx] / stats["hands_played"]) * 100 if stats["hands_played"] > 0 else 0
        print(f"Player {player_idx}: Wins = {stats['player_wins'][player_idx]} ({win_percentage:.2f}%), "
              f"Folds = {stats['hands_folded'][player_idx]} ({fold_percentage:.2f}%)")


def benchmark_simple_game(num_iterations=1000):
    """
    Benchmark a simple game to measure performance.
    
    Args:
        num_iterations: Number of iterations to run
    """
    print(f"Benchmarking simple game ({num_iterations} iterations)...")
    
    # Create a game
    game = pc.SpinGoGame()
    
    # Measure time to play num_iterations hands
    start_time = time.time()
    
    for i in range(num_iterations):
        # Reset for a new hand
        state = game.get_game_state()
        state.reset_for_new_hand()
        state.set_seed(i)  # Use iteration as seed
        
        # Play a hand to completion with random actions
        hand_over = False
        while not hand_over:
            current_player_idx = state.get_current_player_index()
            legal_actions = state.get_legal_actions()
            
            # Choose a random action
            action = random.choice(legal_actions)
            state.apply_action(action)
            
            # Check if hand is over
            hand_over = state.is_hand_over()
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Simple game: {num_iterations / time_taken:.2f} hands/second")
    print(f"Time per hand: {time_taken / num_iterations * 1000:.3f} ms")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run the test
    test_poker_game(num_hands=5, seed=42)
    
    # Run a simple benchmark
    benchmark_simple_game(num_iterations=100)