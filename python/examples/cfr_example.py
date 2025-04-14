import os
import sys
import numpy as np
import time
import argparse
from tqdm import tqdm

# Add the parent directory to the path so we can import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poker_core as pc
from poker.mccfr import MCCFRSolver
from poker.deep_cfr import DeepCFRSolver

def run_mccfr_example(iterations=1000, prune=True, save_path=None):
    """
    Run Monte Carlo CFR on a Spin & Go game.
    
    Args:
        iterations: Number of iterations to run
        prune: Whether to use pruning
        save_path: Path to save the strategy
    """
    # Create a Spin & Go game
    game = pc.SpinGoGame(num_players=3, buy_in=500, small_blind=10, big_blind=20)
    
    # Create an MCCFR solver
    solver = MCCFRSolver(game)
    
    # Run CFR
    print(f"Running {iterations} iterations of Monte Carlo CFR (pruning: {prune})...")
    start_time = time.time()
    solver.cfr(iterations, prune)
    end_time = time.time()
    
    # Print time taken
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Save strategy if a path is provided
    if save_path:
        solver.save_strategy(save_path)
        print(f"Strategy saved to {save_path}")
    
    # Get and print some strategy information
    strategy = solver.get_average_strategy()
    print(f"Total info sets: {len(strategy)}")
    
    # Print a sample of the strategy
    print("\nSample strategies:")
    count = 0
    for key, strat in strategy.items():
        if count < 5:  # Print first 5 info sets
            print(f"Info set: {key}")
            print(f"Strategy: {strat}")
            count += 1
        else:
            break

def run_deep_cfr_example(iterations=10, traversals_per_iter=100, save_path=None):
    """
    Run Deep CFR on a Spin & Go game.
    
    Args:
        iterations: Number of iterations to run
        traversals_per_iter: Number of tree traversals per iteration
        save_path: Path to save the model
    """
    # Create a Spin & Go game
    game = pc.SpinGoGame(num_players=3, buy_in=500, small_blind=10, big_blind=20)
    
    # Create a Deep CFR solver
    solver = DeepCFRSolver(game, hidden_layers=[128, 128])
    
    # Run training
    print(f"Running {iterations} iterations of Deep CFR with {traversals_per_iter} traversals per iteration...")
    start_time = time.time()
    solver.train(iterations, traversals_per_iter)
    end_time = time.time()
    
    # Print time taken
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Save model if a path is provided
    if save_path:
        solver.save_model(save_path)
        print(f"Model saved to {save_path}")
    
    # Evaluate the learned strategy
    print("\nEvaluating strategy...")
    evaluate_deep_cfr_strategy(solver, game)

def evaluate_deep_cfr_strategy(solver, game, num_games=100):
    """
    Evaluate a Deep CFR strategy by playing games.
    
    Args:
        solver: The Deep CFR solver
        game: The game to evaluate on
        num_games: Number of games to play
    """
    # Reset the game
    game = pc.SpinGoGame(
        len(game.get_game_state().get_players()),
        game.get_game_state().get_players()[0].get_stack(),
        game.get_game_state().get_small_blind(),
        game.get_game_state().get_big_blind()
    )
    
    # Play games
    winnings = [0, 0, 0]  # Winnings for each player
    
    for _ in tqdm(range(num_games), desc="Playing games"):
        # Reset game for a new tournament
        game = pc.SpinGoGame(
            len(game.get_game_state().get_players()),
            game.get_game_state().get_players()[0].get_stack(),
            game.get_game_state().get_small_blind(),
            game.get_game_state().get_big_blind()
        )
        
        # Play until the tournament is over
        while not game.is_tournament_over():
            # Deal hole cards
            state = game.get_game_state()
            state.deal_hole_cards()
            
            # Play hand
            while not state.is_hand_over():
                # Get current player
                current_player = state.get_current_player_index()
                
                # Get legal actions
                legal_actions = state.get_legal_actions()
                
                # Get strategy for current player
                strategy = solver.get_strategy(state, current_player)
                
                # Choose action according to strategy
                action_idx = np.random.choice(len(legal_actions), p=strategy)
                action = legal_actions[action_idx]
                
                # Apply action
                state.apply_action(action)
            
            # Record winner of the hand
            if state.is_hand_over():
                state.reset_for_new_hand()
        
        # Record tournament winner
        if game.is_tournament_over():
            winner = game.get_tournament_winner()
            winnings[winner] += game.get_prize_pool()
    
    # Print results
    print("\nTournament results:")
    for i, wins in enumerate(winnings):
        print(f"Player {i}: {wins} chips")

def compare_strategies(num_games=100):
    """
    Compare MCCFR and Deep CFR strategies.
    
    Args:
        num_games: Number of games to play
    """
    # Create a Spin & Go game
    game = pc.SpinGoGame(num_players=2, buy_in=500, small_blind=10, big_blind=20)
    
    # Create solvers
    mccfr_solver = MCCFRSolver(game)
    deep_cfr_solver = DeepCFRSolver(game, hidden_layers=[64, 64])
    
    # Train solvers (minimal training for example purposes)
    print("Training MCCFR solver...")
    mccfr_solver.cfr(100, True)
    
    print("Training Deep CFR solver...")
    deep_cfr_solver.train(5, 20)
    
    # Play games between the strategies
    mccfr_wins = 0
    deep_cfr_wins = 0
    
    for _ in tqdm(range(num_games), desc="Playing comparison games"):
        # Reset game
        game = pc.SpinGoGame(num_players=2, buy_in=500, small_blind=10, big_blind=20)
        
        # Assign strategies to players
        # Player 0 uses MCCFR, Player 1 uses Deep CFR
        
        # Play until the tournament is over
        while not game.is_tournament_over():
            # Deal hole cards
            state = game.get_game_state()
            state.deal_hole_cards()
            
            # Play hand
            while not state.is_hand_over():
                # Get current player
                current_player = state.get_current_player_index()
                
                # Get legal actions
                legal_actions = state.get_legal_actions()
                
                # Choose action according to strategy
                if current_player == 0:
                    # MCCFR strategy
                    info_set_key = mccfr_solver.get_info_set_key(state, current_player)
                    info_set = mccfr_solver.get_info_set(info_set_key, len(legal_actions))
                    strategy = info_set.get_average_strategy()
                    
                    # Ensure strategy is valid
                    if np.sum(strategy) <= 0:
                        strategy = np.ones(len(legal_actions)) / len(legal_actions)
                    else:
                        strategy = strategy / np.sum(strategy)
                else:
                    # Deep CFR strategy
                    strategy = deep_cfr_solver.get_strategy(state, current_player)
                
                # Choose action according to strategy
                action_idx = np.random.choice(len(legal_actions), p=strategy)
                action = legal_actions[action_idx]
                
                # Apply action
                state.apply_action(action)
            
            # Reset for next hand
            if state.is_hand_over():
                state.reset_for_new_hand()
        
        # Record tournament winner
        if game.is_tournament_over():
            winner = game.get_tournament_winner()
            if winner == 0:
                mccfr_wins += 1
            else:
                deep_cfr_wins += 1
    
    # Print results
    print("\nStrategy comparison results:")
    print(f"MCCFR wins: {mccfr_wins} ({mccfr_wins/num_games*100:.1f}%)")
    print(f"Deep CFR wins: {deep_cfr_wins} ({deep_cfr_wins/num_games*100:.1f}%)")

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Poker CFR Examples")
    parser.add_argument("--method", type=str, choices=["mccfr", "deep_cfr", "compare"], 
                        default="mccfr", help="Method to run")
    parser.add_argument("--iterations", type=int, default=1000, 
                        help="Number of iterations for MCCFR")
    parser.add_argument("--deep_iterations", type=int, default=10, 
                        help="Number of iterations for Deep CFR")
    parser.add_argument("--traversals", type=int, default=100, 
                        help="Number of traversals per iteration for Deep CFR")
    parser.add_argument("--save_path", type=str, help="Path to save the strategy or model")
    args = parser.parse_args()
    
    if args.method == "mccfr":
        run_mccfr_example(args.iterations, True, args.save_path)
    elif args.method == "deep_cfr":
        run_deep_cfr_example(args.deep_iterations, args.traversals, args.save_path)
    elif args.method == "compare":
        compare_strategies()

if __name__ == "__main__":
    main()
