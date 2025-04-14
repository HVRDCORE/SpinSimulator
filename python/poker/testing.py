#!/usr/bin/env python3
"""
Advanced testing utilities for poker simulator.
This module provides functions for stress testing, tournament simulation,
and parallel testing capabilities.
"""
import os
import time
import json
import random
import concurrent.futures
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import poker_core as pc
    from poker.logging import GameLogger
    from poker.visualization import PokerVisualizer
except ImportError:
    import sys
    # Try to find the module
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root_dir)
    import poker_core as pc
    from poker.logging import GameLogger
    from poker.visualization import PokerVisualizer

def stress_test_hand_evaluation(num_iterations=100000):
    """
    Stress test the hand evaluation functionality.
    
    Args:
        num_iterations: Number of iterations to run
        
    Returns:
        Dict containing test results
    """
    print(f"Running hand evaluation stress test ({num_iterations} iterations)...")
    
    # Create a hand evaluator
    evaluator = pc.HandEvaluator()
    
    # Create a deck
    deck = pc.Deck()
    
    results = {
        "iterations": num_iterations,
        "start_time": time.time(),
        "hand_types": {
            "HIGH_CARD": 0,
            "PAIR": 0,
            "TWO_PAIR": 0,
            "THREE_OF_A_KIND": 0,
            "STRAIGHT": 0,
            "FLUSH": 0,
            "FULL_HOUSE": 0,
            "FOUR_OF_A_KIND": 0,
            "STRAIGHT_FLUSH": 0,
            "ROYAL_FLUSH": 0
        }
    }
    
    for i in range(num_iterations):
        # Shuffle the deck
        deck.shuffle()
        
        # Deal 7 cards (2 hole cards + 5 community cards)
        cards = [deck.deal_card() for _ in range(7)]
        
        # Evaluate the hand
        result = evaluator.evaluate(cards)
        hand_type = evaluator.get_hand_type(result)
        hand_type_str = str(hand_type)
        
        # Count hand types
        if hand_type_str in results["hand_types"]:
            results["hand_types"][hand_type_str] += 1
    
    # Calculate elapsed time and speed
    elapsed_time = time.time() - results["start_time"]
    results["elapsed_time"] = elapsed_time
    results["hands_per_second"] = num_iterations / elapsed_time
    
    print(f"Hand evaluation stress test completed in {elapsed_time:.2f}s")
    print(f"Speed: {results['hands_per_second']:.2f} hands/second")
    
    # Print distribution of hand types
    print("\nHand type distribution:")
    for hand_type, count in results["hand_types"].items():
        percentage = (count / num_iterations) * 100
        print(f"{hand_type}: {count} ({percentage:.2f}%)")
    
    return results

def run_tournament_simulation(
    num_players=3,
    initial_stack=1000,
    small_blind=10,
    big_blind=20,
    prize_multiplier=2.0,
    max_hands=None,
    log_dir="logs",
    export_format="json",
    visualize=True
):
    """
    Run a complete tournament simulation.
    
    Args:
        num_players: Number of players in the tournament
        initial_stack: Starting stack for each player
        small_blind: Initial small blind
        big_blind: Initial big blind
        prize_multiplier: Multiplier for prize pool
        max_hands: Maximum number of hands to play (None = play until completion)
        log_dir: Directory for logs
        export_format: Format for exporting data (json or csv)
        visualize: Whether to create visualizations
        
    Returns:
        Dict containing tournament results
    """
    print(f"Starting tournament simulation with {num_players} players...")
    
    # Create a logger
    logger = GameLogger(log_dir=log_dir, export_format=export_format)
    
    # Create the game
    game = pc.SpinGoGame(
        num_players=num_players,
        buy_in=initial_stack,
        small_blind=small_blind,
        big_blind=big_blind,
        prize_multiplier=prize_multiplier
    )
    
    # Set a random seed for reproducibility
    seed = int(time.time())
    game.set_seed(seed)
    logger.logger.info(f"Using random seed: {seed}")
    
    # Tournament tracking
    start_time = time.time()
    hands_played = 0
    
    # Define blind schedule (optional)
    game.add_blind_level(small_blind, big_blind, 0)  # Initial level
    game.add_blind_level(small_blind*2, big_blind*2, 20)  # After 20 hands
    game.add_blind_level(small_blind*4, big_blind*4, 40)  # After 40 hands
    
    # Log initial state
    state = game.get_game_state()
    logger.logger.info(f"Tournament starting - Prize pool: {game.get_prize_pool()}")
    logger.logger.info(f"Initial blinds: SB={small_blind}, BB={big_blind}")
    
    # Play until tournament is over or max hands reached
    while not game.is_tournament_over():
        if max_hands and hands_played >= max_hands:
            break
        
        # Get pre-hand state
        old_stage = state.get_current_stage()
        old_blinds = (state.get_small_blind(), state.get_big_blind())
        
        # Remember the action history length before playing
        action_history_before = state.get_action_history()
        
        # Play a hand
        game.play_hand()
        hands_played += 1
        
        # Get post-hand state
        new_stage = state.get_current_stage()
        new_blinds = (state.get_small_blind(), state.get_big_blind())
        
        # Get action history for this hand
        action_history_after = state.get_action_history()
        hand_actions = action_history_after[len(action_history_before):]
        
        # Get winners if hand completed
        winners = game.get_last_hand_winners()
        
        # Log hand completion
        logger.log_hand(
            hand_num=hands_played,
            actions=hand_actions,
            winners=winners if winners else [],
            pot=state.get_pot(),
            players=state.get_players(),
            community_cards=state.get_community_cards()
        )
        
        # Check if blinds changed
        if old_blinds != new_blinds:
            logger.log_blind_change(new_blinds[0], new_blinds[1], hands_played)
        
        # Periodic status update
        if hands_played % 10 == 0:
            active_players = sum(1 for p in state.get_players() if p.get_stack() > 0)
            logger.logger.info(f"Played {hands_played} hands - {active_players} players remaining")
    
    # Tournament completed
    end_time = time.time()
    total_time = end_time - start_time
    
    # Determine winner and rankings
    rankings = []
    winner = -1
    
    for i, player in enumerate(state.get_players()):
        if player.get_stack() > 0:
            winner = i
            rankings.append((i, 1))  # Winner gets rank 1
        else:
            rankings.append((i, 2))  # All others tied for 2nd
    
    # Log tournament completion
    logger.log_tournament_end(
        winner=winner,
        hands_played=hands_played,
        total_time=total_time,
        prize_pool=game.get_prize_pool(),
        player_ranks=rankings
    )
    
    # Export data
    data_file = logger.export_data()
    
    # Create visualizations
    if visualize and data_file:
        vis_dir = os.path.join(log_dir, "visualizations")
        visualizer = PokerVisualizer(data_file, output_dir=vis_dir)
        vis_files = visualizer.create_all_visualizations(show=False)
        logger.logger.info(f"Created {len(vis_files)} visualization files in {vis_dir}")
    
    # Compile results
    results = {
        "tournament_id": logger.session_id,
        "seed": seed,
        "num_players": num_players,
        "prize_pool": game.get_prize_pool(),
        "hands_played": hands_played,
        "total_time": total_time,
        "hands_per_second": hands_played / total_time if total_time > 0 else 0,
        "winner": winner,
        "player_ranks": rankings,
        "data_file": data_file
    }
    
    print(f"Tournament simulation completed in {total_time:.2f}s")
    print(f"Winner: Player {winner}")
    print(f"Hands played: {hands_played}")
    print(f"Hands per second: {results['hands_per_second']:.2f}")
    
    return results

def parallel_tournament_simulations(
    num_tournaments=10,
    num_threads=4,
    **tournament_kwargs
):
    """
    Run multiple tournament simulations in parallel.
    
    Args:
        num_tournaments: Total number of tournaments to simulate
        num_threads: Number of parallel threads to use
        **tournament_kwargs: Arguments to pass to run_tournament_simulation
        
    Returns:
        List of results from all tournaments
    """
    print(f"Running {num_tournaments} parallel tournament simulations using {num_threads} threads...")
    
    # Create a base log directory
    base_log_dir = tournament_kwargs.get("log_dir", "logs")
    
    results = []
    
    def run_single_tournament(tournament_idx):
        """Run a single tournament with unique log directory"""
        # Create a unique log directory for this tournament
        log_dir = os.path.join(base_log_dir, f"tournament_{tournament_idx}")
        kwargs = tournament_kwargs.copy()
        kwargs["log_dir"] = log_dir
        
        # Run the tournament
        try:
            result = run_tournament_simulation(**kwargs)
            return result
        except Exception as e:
            print(f"Error in tournament {tournament_idx}: {str(e)}")
            return {"error": str(e), "tournament_idx": tournament_idx}
    
    # Run tournaments in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {
            executor.submit(run_single_tournament, i): i 
            for i in range(num_tournaments)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Tournament {idx} completed")
            except Exception as e:
                print(f"Tournament {idx} failed: {str(e)}")
                results.append({"error": str(e), "tournament_idx": idx})
    
    # Analyze results
    successful = len([r for r in results if "error" not in r])
    print(f"Completed {successful}/{num_tournaments} tournaments successfully")
    
    # Calculate aggregate statistics
    if successful > 0:
        total_hands = sum(r.get("hands_played", 0) for r in results if "error" not in r)
        total_time = sum(r.get("total_time", 0) for r in results if "error" not in r)
        
        print(f"Total hands played across all tournaments: {total_hands}")
        print(f"Average hands per tournament: {total_hands / successful:.2f}")
        print(f"Average hands per second: {total_hands / total_time:.2f}")
        
        # Count winners
        winner_counts = {}
        for r in results:
            if "error" not in r and "winner" in r:
                winner = r["winner"]
                winner_counts[winner] = winner_counts.get(winner, 0) + 1
        
        print("\nWinner distribution:")
        for player, count in sorted(winner_counts.items()):
            percentage = (count / successful) * 100
            print(f"Player {player}: {count} wins ({percentage:.2f}%)")
    
    return results

def test_hand_evaluator_accuracy():
    """
    Test the accuracy of hand evaluation by checking known hands.
    
    Returns:
        A dictionary with test results
    """
    print("Testing hand evaluator accuracy...")
    
    # Create a hand evaluator
    evaluator = pc.HandEvaluator()
    
    # Define test cases with expected outcomes
    # Format: (cards, expected_type)
    test_cases = [
        # Royal Flush
        (["AH", "KH", "QH", "JH", "TH", "2S", "3C"], pc.HandType.ROYAL_FLUSH),
        
        # Straight Flush
        (["9C", "8C", "7C", "6C", "5C", "KD", "AH"], pc.HandType.STRAIGHT_FLUSH),
        
        # Four of a Kind
        (["8H", "8D", "8S", "8C", "4D", "4H", "7S"], pc.HandType.FOUR_OF_A_KIND),
        
        # Full House
        (["3H", "3D", "3S", "9C", "9D", "2S", "5C"], pc.HandType.FULL_HOUSE),
        
        # Flush
        (["AH", "JH", "9H", "6H", "2H", "KC", "QD"], pc.HandType.FLUSH),
        
        # Straight
        (["8D", "7C", "6H", "5S", "4D", "KH", "AC"], pc.HandType.STRAIGHT),
        
        # Three of a Kind
        (["7H", "7D", "7S", "KD", "3C", "JH", "TD"], pc.HandType.THREE_OF_A_KIND),
        
        # Two Pair
        (["JH", "JD", "4S", "4C", "AD", "8H", "3S"], pc.HandType.TWO_PAIR),
        
        # Pair
        (["TH", "TD", "AH", "KS", "QC", "5D", "2C"], pc.HandType.PAIR),
        
        # High Card
        (["AH", "JD", "9S", "7C", "5D", "3H", "2S"], pc.HandType.HIGH_CARD),
    ]
    
    results = {
        "total_cases": len(test_cases),
        "passed": 0,
        "failed": 0,
        "failures": []
    }
    
    for i, (card_strings, expected_type) in enumerate(test_cases):
        # Convert strings to Card objects
        cards = [pc.Card.from_string(card_str) for card_str in card_strings]
        
        # Evaluate the hand
        hand_value = evaluator.evaluate(cards)
        hand_type = evaluator.get_hand_type(hand_value)
        
        # Check if the result matches the expected type
        if hand_type == expected_type:
            results["passed"] += 1
            print(f"Test case {i+1}: PASSED - {card_strings} → {hand_type}")
        else:
            results["failed"] += 1
            error = f"Expected {expected_type}, got {hand_type}"
            results["failures"].append({
                "case": i+1,
                "cards": card_strings,
                "expected": str(expected_type),
                "actual": str(hand_type),
                "error": error
            })
            print(f"Test case {i+1}: FAILED - {card_strings} → {error}")
    
    # Print summary
    print(f"\nSummary: {results['passed']}/{results['total_cases']} test cases passed")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced poker simulator testing utilities")
    parser.add_argument("--test", choices=["hand_evaluator", "tournament", "parallel", "accuracy"], 
                        required=True, help="Test type to run")
    parser.add_argument("--iterations", type=int, default=10000, 
                        help="Number of iterations for tests")
    parser.add_argument("--threads", type=int, default=4, 
                        help="Number of threads for parallel tests")
    parser.add_argument("--players", type=int, default=3,
                        help="Number of players for tournament tests")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations for tournament tests")
    
    args = parser.parse_args()
    
    if args.test == "hand_evaluator":
        stress_test_hand_evaluation(args.iterations)
    
    elif args.test == "tournament":
        run_tournament_simulation(
            num_players=args.players,
            max_hands=args.iterations,
            visualize=args.visualize
        )
    
    elif args.test == "parallel":
        parallel_tournament_simulations(
            num_tournaments=args.iterations,
            num_threads=args.threads,
            num_players=args.players,
            max_hands=100,  # Shorter tournaments for parallel testing
            visualize=args.visualize
        )
    
    elif args.test == "accuracy":
        test_hand_evaluator_accuracy()