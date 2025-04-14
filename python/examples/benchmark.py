import os
import sys
import time
import random
import numpy as np
import argparse
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

def benchmark_hand_evaluation(num_iterations=100000):
    """
    Benchmark the performance of hand evaluation.
    
    Args:
        num_iterations: Number of iterations to run
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

def benchmark_game_simulation(num_iterations=10000):
    """
    Benchmark the performance of game simulation.
    
    Args:
        num_iterations: Number of iterations to run
    """
    print(f"Benchmarking game simulation ({num_iterations} iterations)...")
    
    # Create a game
    game = pc.SpinGoGame(num_players=3, buy_in=500, small_blind=10, big_blind=20)
    
    # Simulate hands
    start_time = time.time()
    
    for _ in tqdm(range(num_iterations)):
        # Reset game
        game = pc.SpinGoGame(num_players=3, buy_in=500, small_blind=10, big_blind=20)
        
        # Deal hole cards
        game.get_game_state().deal_hole_cards()
        
        # Play hand
        while not game.get_game_state().is_hand_over():
            # Get legal actions
            state = game.get_game_state()
            legal_actions = state.get_legal_actions()
            
            # Choose random action
            action = legal_actions[random.randint(0, len(legal_actions) - 1)]
            
            # Apply action
            state.apply_action(action)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Game simulation: {num_iterations / time_taken:.2f} hands/second")
    print(f"Time per hand: {time_taken / num_iterations * 1000:.3f} ms")

def benchmark_tournament_simulation(num_iterations=100):
    """
    Benchmark the performance of tournament simulation.
    
    Args:
        num_iterations: Number of iterations to run
    """
    print(f"Benchmarking tournament simulation ({num_iterations} iterations)...")
    
    # Simulate tournaments
    start_time = time.time()
    
    for _ in tqdm(range(num_iterations)):
        # Create a game
        game = pc.SpinGoGame(num_players=3, buy_in=500, small_blind=10, big_blind=20)
        
        # Play until the tournament is over
        while not game.is_tournament_over():
            # Deal hole cards
            game.get_game_state().deal_hole_cards()
            
            # Play hand
            while not game.get_game_state().is_hand_over():
                # Get legal actions
                state = game.get_game_state()
                legal_actions = state.get_legal_actions()
                
                # Choose random action
                action = legal_actions[random.randint(0, len(legal_actions) - 1)]
                
                # Apply action
                state.apply_action(action)
            
            # Reset for next hand
            if game.get_game_state().is_hand_over():
                game.get_game_state().reset_for_new_hand()
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Tournament simulation: {num_iterations / time_taken:.2f} tournaments/second")
    print(f"Time per tournament: {time_taken / num_iterations:.3f} seconds")

def benchmark_python_bindings(num_iterations=10000):
    """
    Benchmark the performance of Python bindings.
    
    Args:
        num_iterations: Number of iterations to run
    """
    print(f"Benchmarking Python bindings ({num_iterations} iterations)...")
    
    # Create cards
    start_time = time.time()
    
    for _ in tqdm(range(num_iterations)):
        # Create 52 cards
        cards = []
        for i in range(52):
            cards.append(pc.Card(i))
        
        # Create a deck
        deck = pc.Deck(cards)
        
        # Shuffle deck - use a seed value directly
        seed = random.randint(0, 2**31-1)
        deck.shuffle(seed)
        
        # Deal 10 cards
        for _ in range(10):
            deck.deal_card()
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    print(f"Python bindings: {num_iterations / time_taken:.2f} iterations/second")
    print(f"Time per iteration: {time_taken / num_iterations * 1000:.3f} ms")

def benchmark_all():
    """
    Run all benchmarks.
    """
    print("Running all benchmarks...")
    print("-" * 50)
    
    # Hand evaluation
    benchmark_hand_evaluation()
    print("-" * 50)
    
    # Game simulation
    benchmark_game_simulation()
    print("-" * 50)
    
    # Tournament simulation
    benchmark_tournament_simulation()
    print("-" * 50)
    
    # Python bindings
    benchmark_python_bindings()
    print("-" * 50)

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Poker Benchmarks")
    parser.add_argument("--benchmark", type=str, 
                        choices=["hand_evaluation", "game_simulation", 
                                "tournament_simulation", "python_bindings", "all"], 
                        default="all", help="Benchmark to run")
    parser.add_argument("--iterations", type=int, 
                        help="Number of iterations to run")
    args = parser.parse_args()
    
    # Set default iterations for each benchmark
    if not args.iterations:
        if args.benchmark == "hand_evaluation":
            args.iterations = 100000
        elif args.benchmark == "game_simulation":
            args.iterations = 10000
        elif args.benchmark == "tournament_simulation":
            args.iterations = 100
        elif args.benchmark == "python_bindings":
            args.iterations = 10000
    
    # Run the selected benchmark
    if args.benchmark == "hand_evaluation":
        benchmark_hand_evaluation(args.iterations)
    elif args.benchmark == "game_simulation":
        benchmark_game_simulation(args.iterations)
    elif args.benchmark == "tournament_simulation":
        benchmark_tournament_simulation(args.iterations)
    elif args.benchmark == "python_bindings":
        benchmark_python_bindings(args.iterations)
    elif args.benchmark == "all":
        benchmark_all()

if __name__ == "__main__":
    main()
