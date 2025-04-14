#!/usr/bin/env python3
"""
Quick benchmark of key poker simulator components.
This script runs a limited set of benchmarks to quickly assess performance.
"""
import os
import sys
import time
import random
from tqdm import tqdm

# Add the parent directories to the path so we can import our module
examples_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(examples_dir)
project_dir = os.path.dirname(python_dir)
sys.path.append(python_dir)
sys.path.append(project_dir)  # Add root directory too, for direct imports

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

def quick_hand_evaluation(num_iterations=1000):
    """
    Quick benchmark of hand evaluation performance.
    """
    print(f"Benchmarking hand evaluation ({num_iterations} iterations)...")
    
    # Create a hand evaluator
    evaluator = pc.HandEvaluator()
    
    # Create a deck
    deck = pc.Deck()
    
    # Generate random hands
    start_time = time.time()
    
    for _ in tqdm(range(num_iterations)):
        # Reset deck
        deck.reset()
        
        # Shuffle deck - use a seed value directly
        seed = random.randint(0, 2**31-1)
        deck.shuffle(seed)
        
        # Deal 7 cards
        cards = []
        for _ in range(7):
            cards.append(deck.deal_card())
        
        # Evaluate hand
        evaluator.evaluate(cards)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Hand evaluation: {num_iterations / time_taken:.2f} hands/second")
    print(f"Time per hand: {time_taken / num_iterations * 1000:.3f} ms")
    return num_iterations / time_taken

def quick_game_simulation(num_iterations=100):
    """
    Quick benchmark of game simulation performance.
    """
    print(f"Benchmarking game simulation ({num_iterations} iterations)...")
    
    # Simulate hands
    start_time = time.time()
    
    for _ in tqdm(range(num_iterations)):
        # Create a new game
        game = pc.SpinGoGame(num_players=3, buy_in=500, small_blind=10, big_blind=20)
        
        # Deal hole cards
        game.get_game_state().deal_hole_cards()
        
        # Play one round of betting
        state = game.get_game_state()
        for _ in range(10):  # Limit to a small number of actions
            if state.is_hand_over():
                break
                
            # Get legal actions
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                break
                
            # Choose random action
            action = legal_actions[random.randint(0, len(legal_actions) - 1)]
            
            # Apply action
            state.apply_action(action)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Game simulation: {num_iterations / time_taken:.2f} games/second")
    print(f"Time per game: {time_taken / num_iterations * 1000:.3f} ms")
    return num_iterations / time_taken

def quick_python_bindings(num_iterations=1000):
    """
    Quick benchmark of Python bindings performance.
    """
    print(f"Benchmarking Python bindings ({num_iterations} iterations)...")
    
    # Create cards
    start_time = time.time()
    
    for _ in tqdm(range(num_iterations)):
        # Create a deck
        deck = pc.Deck()
        
        # Shuffle deck - use a seed value directly
        seed = random.randint(0, 2**31-1)
        deck.shuffle(seed)
        
        # Deal 5 cards
        for _ in range(5):
            deck.deal_card()
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Python bindings: {num_iterations / time_taken:.2f} iterations/second")
    print(f"Time per iteration: {time_taken / num_iterations * 1000:.3f} ms")
    return num_iterations / time_taken

def quick_simple_game_test():
    """
    Run a simple game test to verify functionality.
    """
    print("Running a simple game test...")
    
    # Create a SpinGoGame
    game = pc.SpinGoGame(num_players=3, buy_in=500, small_blind=10, big_blind=20)
    
    # Set a seed for reproducibility
    game.set_seed(42)
    
    # Deal hole cards
    state = game.get_game_state()
    state.deal_hole_cards()
    
    # Print initial state
    print("Initial state:")
    for i in range(3):
        player = state.get_players()[i]
        hole_cards = player.get_hole_cards()
        hole_cards_str = ", ".join([card.to_string() for card in hole_cards])
        print(f"Player {i}: Cards={hole_cards_str}, Stack={player.get_stack()}")
    
    # Play a few random actions
    for _ in range(10):
        if state.is_hand_over():
            break
            
        current_player_idx = state.get_current_player_index()
        legal_actions = state.get_legal_actions()
        
        if not legal_actions:
            break
            
        # Choose random action
        action = legal_actions[random.randint(0, len(legal_actions) - 1)]
        print(f"Player {current_player_idx} performs: {action.to_string()}")
        
        # Apply action
        state.apply_action(action)
    
    # Print final state
    print("Final state:")
    for i in range(3):
        player = state.get_players()[i]
        print(f"Player {i}: Stack={player.get_stack()}, Folded={player.has_folded()}")
    
    # Show winners if hand is over
    if state.is_hand_over():
        winners = state.get_winners()
        if winners:
            for winner_idx in winners:
                print(f"Player {winner_idx} won with: {state.get_hand_description(winner_idx)}")
        else:
            print("No winners")

def main():
    """
    Run quick benchmarks to test performance.
    """
    print("Running quick benchmarks of poker simulator...")
    print("-" * 50)
    
    # Run quick hand evaluation benchmark
    hand_eval_speed = quick_hand_evaluation()
    print("-" * 50)
    
    # Run quick game simulation benchmark
    game_sim_speed = quick_game_simulation()
    print("-" * 50)
    
    # Run quick Python bindings benchmark
    binding_speed = quick_python_bindings()
    print("-" * 50)
    
    # Run simple game test
    quick_simple_game_test()
    print("-" * 50)
    
    # Summary
    print("Performance Summary:")
    print(f"Hand evaluation: {hand_eval_speed:.2f} hands/second")
    print(f"Game simulation: {game_sim_speed:.2f} games/second")
    print(f"Python bindings: {binding_speed:.2f} iterations/second")
    
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()