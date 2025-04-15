#!/usr/bin/env python3
"""
Simple text-based visualization for poker games without external dependencies.
This is a stripped-down version that doesn't require matplotlib, seaborn, or TensorFlow.
"""
import os
import sys
import json
import time
import random
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

# Simplified version of the GameLogger class that doesn't import from poker.logging
class SimpleTextLogger:
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

# Simplified text-based visualizer
class TextVisualizer:
    """Simple text-based visualizer that generates reports from poker data."""
    
    def __init__(self, data_source, output_dir="text_reports"):
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
    
    def create_text_bankroll_report(self, save_path=None):
        """Create a text-based report of player bankrolls over time."""
        if not save_path:
            save_path = os.path.join(self.output_dir, "bankroll_report.txt")
        
        with open(save_path, 'w') as f:
            f.write("=== PLAYER BANKROLL REPORT ===\n\n")
            f.write("Hand | " + " | ".join([f"Player {i}" for i in sorted(self.data["player_bankrolls"].keys())]) + "\n")
            f.write("-" * 80 + "\n")
            
            # Find the maximum number of hands
            max_hands = max(len(rolls) for rolls in self.data["player_bankrolls"].values())
            
            # Write bankroll data for each hand
            for hand_idx in range(max_hands):
                hand_data = [str(hand_idx + 1)]
                
                for player_id in sorted(self.data["player_bankrolls"].keys()):
                    if hand_idx < len(self.data["player_bankrolls"][player_id]):
                        hand_data.append(str(self.data["player_bankrolls"][player_id][hand_idx]))
                    else:
                        hand_data.append("N/A")
                
                f.write(" | ".join(hand_data) + "\n")
            
            # Add summary
            f.write("\n=== SUMMARY ===\n")
            f.write("Final stacks:\n")
            for player_id in sorted(self.data["player_bankrolls"].keys()):
                bankroll = self.data["player_bankrolls"][player_id]
                if bankroll:
                    f.write(f"Player {player_id}: {bankroll[-1]}\n")
            
            # Calculate total change
            f.write("\nStack changes from initial:\n")
            for player_id in sorted(self.data["player_bankrolls"].keys()):
                bankroll = self.data["player_bankrolls"][player_id]
                if len(bankroll) > 1:
                    change = bankroll[-1] - bankroll[0]
                    f.write(f"Player {player_id}: {'+' if change >= 0 else ''}{change}\n")
        
        print(f"Bankroll report saved to {save_path}")
        return save_path
    
    def create_text_pot_report(self, save_path=None):
        """Create a text-based report of pot sizes across hands."""
        if not save_path:
            save_path = os.path.join(self.output_dir, "pot_report.txt")
        
        with open(save_path, 'w') as f:
            f.write("=== POT SIZE REPORT ===\n\n")
            f.write("Hand | Pot Size\n")
            f.write("-" * 30 + "\n")
            
            # Extract pot sizes from hands
            pot_sizes = [hand["pot"] for hand in self.data["hands"]]
            
            # Write pot size data for each hand
            for i, pot in enumerate(pot_sizes):
                f.write(f"{i+1} | {pot}\n")
            
            # Add summary statistics
            if pot_sizes:
                f.write("\n=== SUMMARY ===\n")
                f.write(f"Average pot size: {sum(pot_sizes) / len(pot_sizes):.2f}\n")
                f.write(f"Largest pot: {max(pot_sizes)}\n")
                f.write(f"Smallest pot: {min(pot_sizes)}\n")
        
        print(f"Pot report saved to {save_path}")
        return save_path
    
    def create_text_tournament_summary(self, save_path=None):
        """Create a text-based summary of the tournament."""
        if not self.data.get("tournament_stats"):
            return ""
            
        if not save_path:
            save_path = os.path.join(self.output_dir, "tournament_summary.txt")
        
        stats = self.data["tournament_stats"]
        with open(save_path, 'w') as f:
            f.write("=== TOURNAMENT SUMMARY ===\n\n")
            
            # Tournament statistics
            f.write("Tournament Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Hands Played: {stats['hands_played']}\n")
            f.write(f"Total Time: {stats['total_time']:.2f} seconds\n")
            hands_per_sec = stats['hands_played'] / stats['total_time'] if stats['total_time'] > 0 else 0
            f.write(f"Hands Per Second: {hands_per_sec:.2f}\n")
            f.write(f"Prize Pool: {stats['prize_pool']}\n")
            f.write(f"Winner: Player {stats['winner']}\n\n")
            
            # Win stats
            f.write("Win Statistics:\n")
            f.write("-" * 30 + "\n")
            win_counts = {}
            for winner_data in self.data["winners"]:
                player_id = winner_data["player"]
                if player_id not in win_counts:
                    win_counts[player_id] = 0
                win_counts[player_id] += 1
            
            total_hands = len(self.data["hands"])
            for player_id, wins in sorted(win_counts.items()):
                win_pct = (wins / total_hands) * 100
                f.write(f"Player {player_id}: {wins} wins ({win_pct:.2f}%)\n")
            
            # Blind progression
            if self.data.get("blinds_history"):
                f.write("\nBlind Progression:\n")
                f.write("-" * 30 + "\n")
                f.write("Hand | Small Blind | Big Blind\n")
                previous_hand = 0
                for blind_change in self.data["blinds_history"]:
                    hand = blind_change["hand"]
                    sb = blind_change["small_blind"]
                    bb = blind_change["big_blind"]
                    if previous_hand == 0:
                        f.write(f"1-{hand} | {sb} | {bb}\n")
                    else:
                        f.write(f"{previous_hand+1}-{hand} | {sb} | {bb}\n")
                    previous_hand = hand
        
        print(f"Tournament summary saved to {save_path}")
        return save_path
    
    def create_all_text_reports(self):
        """Create all available text reports."""
        paths = []
        
        # Create basic reports
        paths.append(self.create_text_bankroll_report())
        paths.append(self.create_text_pot_report())
        
        # Create tournament summary if available
        summary_path = self.create_text_tournament_summary()
        if summary_path:
            paths.append(summary_path)
        
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
    logger = SimpleTextLogger(log_dir=log_dir, export_format="json")
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

def create_text_reports(data_file, output_dir="./text_reports"):
    """
    Create text reports from the tournament data.
    
    Args:
        data_file: Path to the exported JSON data file
        output_dir: Directory to store text reports
    """
    print(f"Creating text reports from {data_file}...")
    
    # Create the visualizer
    visualizer = TextVisualizer(data_file, output_dir=output_dir)
    
    # Create all text reports
    report_files = visualizer.create_all_text_reports()
    
    print(f"Created {len(report_files)} report files in {output_dir}")
    print("Report files:")
    for file in report_files:
        print(f"  - {os.path.basename(file)}")
    
    return report_files

def main():
    """
    Run the demonstration of logging and text reporting without graphical dependencies.
    """
    # Create directories
    log_dir = os.path.join(project_dir, "logs")
    report_dir = os.path.join(log_dir, "text_reports")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    # Play a logged tournament
    data_file = play_logged_tournament(num_hands=20, log_dir=log_dir)
    
    # Create text reports
    if data_file:
        create_text_reports(data_file, output_dir=report_dir)
    
if __name__ == "__main__":
    random.seed(42)  # Set random seed for reproducibility
    main()