#!/usr/bin/env python3
"""
Simple visualization for poker games without TensorFlow dependencies.
This version avoids importing deep_cfr module which requires TensorFlow.
"""
import os
import sys
import json
import time
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Simplified version of the GameLogger class that doesn't import from poker.logging
class SimpleGameLogger:
    """Simplified version of GameLogger that doesn't have external dependencies."""
    
    def __init__(self, log_dir="logs", export_format="json"):
        self.log_dir = log_dir
        self.export_format = export_format
        os.makedirs(log_dir, exist_ok=True)
        
        # Init data structure for collecting game information
        self.history = {
            "session_id": time.strftime("%Y%m%d_%H%M%S"),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hands": [],
            "player_bankrolls": {},
            "blinds_history": [],
            "winners": [],
            "tournament_stats": {}
        }
    
    def log_hand(self, hand_num, actions, winners, pot, players, community_cards=None):
        """Log information about a completed hand."""
        # Convert actions to serializable format
        action_data = []
        for action in actions:
            action_data.append({
                "type": str(action.get_type()),
                "amount": action.get_amount()
            })
        
        # Record community cards
        card_data = []
        if community_cards:
            for card in community_cards:
                card_data.append(card.to_string())
        
        # Collect hand data
        hand_data = {
            "hand_num": hand_num,
            "actions": action_data,
            "winners": winners,
            "pot": pot,
            "player_stacks": {},
            "community_cards": card_data
        }
        
        # Record player stacks
        for i, player in enumerate(players):
            hand_data["player_stacks"][i] = player.get_stack()
            
            # Update bankroll history
            if i not in self.history["player_bankrolls"]:
                self.history["player_bankrolls"][i] = []
            self.history["player_bankrolls"][i].append(player.get_stack())
        
        # Add the hand data to history
        self.history["hands"].append(hand_data)
        
        # Record winners
        for winner in winners:
            self.history["winners"].append({
                "hand": hand_num,
                "player": winner,
                "pot": pot
            })
    
    def log_blind_change(self, small_blind, big_blind, hand_num):
        """Log a change in blind levels."""
        self.history["blinds_history"].append({
            "hand": hand_num,
            "small_blind": small_blind,
            "big_blind": big_blind
        })
    
    def log_tournament_end(self, winner, hands_played, total_time, prize_pool, player_ranks=None):
        """Log the end of a tournament."""
        self.history["tournament_stats"] = {
            "winner": winner,
            "hands_played": hands_played,
            "total_time": total_time,
            "prize_pool": prize_pool,
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "player_ranks": player_ranks or []
        }
    
    def log_action(self, player_idx, action):
        """Log a single player action."""
        # Just a placeholder for API compatibility
        pass
    
    def export_data(self, filename=None):
        """Export collected data to a file."""
        if filename is None:
            filename = f"poker_sim_{self.history['session_id']}"
        
        filepath = os.path.join(self.log_dir, filename)
        
        if self.export_format == "json":
            with open(f"{filepath}.json", "w") as f:
                json.dump(self.history, f, indent=4)
            print(f"Exported data to {filepath}.json")
            return f"{filepath}.json"
        else:
            print(f"Unsupported export format: {self.export_format}")
            return ""

# Simplified version of the PokerVisualizer class
class SimplePokerVisualizer:
    """Simplified version of PokerVisualizer that doesn't have external dependencies."""
    
    def __init__(self, data_source, output_dir="visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        if isinstance(data_source, str):
            with open(data_source, 'r') as f:
                if data_source.endswith('.json'):
                    self.data = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {data_source}")
        else:
            self.data = data_source
        
        # Configure plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Default color palette
        self.colors = sns.color_palette("viridis", 10)
    
    def plot_bankrolls(self, save_path=None, show=True):
        """Plot the evolution of each player's bankroll over time."""
        if not save_path:
            save_path = os.path.join(self.output_dir, "bankroll_history.png")
        
        plt.figure(figsize=(12, 8))
        
        for player_idx, bankroll in self.data["player_bankrolls"].items():
            player_name = f"Player {player_idx}"
            plt.plot(bankroll, label=player_name, color=self.colors[int(player_idx) % len(self.colors)])
        
        plt.title("Player Bankrolls Over Time", fontsize=16)
        plt.xlabel("Hand Number", fontsize=14)
        plt.ylabel("Stack Size", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_pot_sizes(self, save_path=None, show=True):
        """Plot the evolution of pot sizes across hands."""
        if not save_path:
            save_path = os.path.join(self.output_dir, "pot_sizes.png")
        
        plt.figure(figsize=(12, 8))
        
        # Extract pot sizes from hands
        pot_sizes = [hand["pot"] for hand in self.data["hands"]]
        hand_numbers = list(range(1, len(pot_sizes) + 1))
        
        plt.plot(hand_numbers, pot_sizes, marker='o', linestyle='-', markersize=6, alpha=0.7)
        
        plt.title("Pot Sizes Over Time", fontsize=16)
        plt.xlabel("Hand Number", fontsize=14)
        plt.ylabel("Pot Size", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add a trend line
        try:
            # Only add trend if we have enough hands
            if len(pot_sizes) > 5:
                z = np.polyfit(hand_numbers, pot_sizes, 1)
                p = np.poly1d(z)
                plt.plot(hand_numbers, p(hand_numbers), "r--", alpha=0.7,
                        label=f"Trend: {z[0]:.2f}x + {z[1]:.2f}")
                plt.legend(fontsize=12)
        except:
            # If trend line fails (e.g., constant values), continue without it
            pass
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def create_all_visualizations(self, show=False):
        """Create all available visualizations."""
        paths = []
        
        # Create basic plots
        paths.append(self.plot_bankrolls(show=show))
        paths.append(self.plot_pot_sizes(show=show))
        
        return paths

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

def play_logged_tournament(num_hands=20, num_players=3, log_dir="./logs"):
    """
    Play a tournament with full logging and visualization.
    
    Args:
        num_hands: Maximum number of hands to play
        num_players: Number of players in the tournament
        log_dir: Directory to store logs and visualizations
    
    Returns:
        Path to the exported data file
    """
    print(f"Playing a logged tournament with {num_players} players (max {num_hands} hands)...")
    
    # Create logger
    logger = SimpleGameLogger(log_dir=log_dir, export_format="json")
    print(f"Logging to {log_dir}")
    
    # Create the game
    game = pc.SpinGoGame(
        num_players=num_players,
        buy_in=1000,
        small_blind=10,
        big_blind=20,
        prize_multiplier=2.0
    )
    
    # Set a fixed seed for reproducibility
    game.set_seed(42)
    
    # Define blind schedule
    game.add_blind_level(10, 20, 0)    # Initial blinds
    game.add_blind_level(20, 40, 5)    # After 5 hands
    game.add_blind_level(30, 60, 10)   # After 10 hands
    game.add_blind_level(50, 100, 15)  # After 15 hands
    
    state = game.get_game_state()
    
    # Initialize statistics
    hands_played = 0
    start_time = time.time()
    
    print(f"Tournament starting - Prize pool: {game.get_prize_pool()}")
    
    # Play hands
    while hands_played < num_hands and not game.is_tournament_over():
        # Record the action history before the hand
        pre_hand_actions = state.get_action_history().copy() if hasattr(state.get_action_history(), 'copy') else []
        pre_hand_blinds = (state.get_small_blind(), state.get_big_blind())
        
        # Play a single hand
        print(f"\n----- Hand #{hands_played + 1} -----")
        print(f"Blinds: SB={state.get_small_blind()}, BB={state.get_big_blind()}")
        
        # Deal hole cards
        state.reset_for_new_hand()
        state.deal_hole_cards()
        
        # Show initial state
        for i in range(num_players):
            player = state.get_players()[i]
            hole_cards = player.get_hole_cards()
            hole_cards_str = ", ".join([card.to_string() for card in hole_cards])
            print(f"Player {i}: Cards={hole_cards_str}, Stack={player.get_stack()}")
        
        # Play the hand
        hand_over = False
        action_count = 0
        
        while not hand_over and action_count < 100:  # Safety limit on actions
            current_player_idx = state.get_current_player_index()
            legal_actions = state.get_legal_actions()
            
            if not legal_actions:
                break
            
            # Choose action using the heuristic agent
            action = simple_heuristic_agent(legal_actions, current_player_idx, state)
            
            # Log the action
            logger.log_action(current_player_idx, action)
            
            print(f"Player {current_player_idx} performs: {action.to_string()}")
            
            # Apply the action
            state.apply_action(action)
            action_count += 1
            
            # Check if hand is over
            if state.is_hand_over():
                hand_over = True
        
        # Get the action history after the hand
        post_hand_actions = state.get_action_history()
        hand_actions = post_hand_actions[len(pre_hand_actions):] if pre_hand_actions else post_hand_actions
        
        # Get winners and pot
        winners = []
        pot = state.get_pot()
        
        # Determine winners if available
        last_winners = state.get_winners()
        if last_winners:
            winners = last_winners
            print(f"Winners: {winners}, Pot: {pot}")
        
        # Log the hand
        logger.log_hand(
            hand_num=hands_played + 1,
            actions=hand_actions,
            winners=winners,
            pot=pot,
            players=state.get_players(),
            community_cards=state.get_community_cards()
        )
        
        # Check if blinds changed
        post_hand_blinds = (state.get_small_blind(), state.get_big_blind())
        if pre_hand_blinds != post_hand_blinds:
            logger.log_blind_change(post_hand_blinds[0], post_hand_blinds[1], hands_played + 1)
            print(f"Blinds increased to SB={post_hand_blinds[0]}, BB={post_hand_blinds[1]}")
        
        hands_played += 1
    
    # Tournament completed
    end_time = time.time()
    total_time = end_time - start_time
    
    # Determine the winner
    winner = -1
    for i, player in enumerate(state.get_players()):
        if player.get_stack() > 0:
            winner = i
            break
    
    # Log tournament results
    rankings = [(i, 1 if i == winner else 2) for i in range(num_players)]
    logger.log_tournament_end(
        winner=winner,
        hands_played=hands_played,
        total_time=total_time,
        prize_pool=game.get_prize_pool(),
        player_ranks=rankings
    )
    
    print(f"\nTournament completed in {total_time:.2f}s")
    print(f"Winner: Player {winner}")
    print(f"Hands played: {hands_played}")
    
    # Export data
    data_file = logger.export_data()
    print(f"Data exported to {data_file}")
    
    return data_file

def create_visualizations(data_file, output_dir="./visualizations"):
    """
    Create visualizations from the tournament data.
    
    Args:
        data_file: Path to the exported JSON data file
        output_dir: Directory to store visualizations
    """
    print(f"Creating visualizations from {data_file}...")
    
    # Create the visualizer
    visualizer = SimplePokerVisualizer(data_file, output_dir=output_dir)
    
    # Create all visualizations
    vis_files = visualizer.create_all_visualizations(show=False)
    
    print(f"Created {len(vis_files)} visualization files in {output_dir}")
    print("Visualization files:")
    for file in vis_files:
        print(f"  - {os.path.basename(file)}")
    
    return vis_files

def main():
    """
    Run the demonstration of logging and visualization without TensorFlow dependencies.
    """
    # Create directories
    log_dir = os.path.join(project_dir, "logs")
    vis_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Play a logged tournament
    data_file = play_logged_tournament(num_hands=20, log_dir=log_dir)
    
    # Create visualizations
    if data_file:
        create_visualizations(data_file, output_dir=vis_dir)
    
if __name__ == "__main__":
    random.seed(42)  # Set random seed for reproducibility
    main()