#!/usr/bin/env python3
"""
Enhanced logging functionality for poker simulations.
This module provides structured logging capabilities and data collection
for later analysis and visualization.
"""
import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Setup standard Python logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class GameLogger:
    """
    Advanced logger for poker games that captures detailed information
    about hands, actions, and player performances.
    """
    
    def __init__(
        self, 
        log_dir: str = "logs", 
        log_level: str = "INFO",
        export_format: str = "json",
        session_id: Optional[str] = None
    ):
        """
        Initialize the game logger.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            export_format: Format for data export (json or csv)
            session_id: Unique identifier for this session (defaults to timestamp)
        """
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        self.export_format = export_format.lower()
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize the logger
        self.logger = logging.getLogger(f"poker_sim_{self.session_id}")
        self.logger.setLevel(self.log_level)
        
        # Add file handler
        log_file = os.path.join(log_dir, f"poker_sim_{self.session_id}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
        
        # Data collection for analysis
        self.history = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "hands": [],
            "player_bankrolls": {},
            "blinds_history": [],
            "winners": [],
            "tournament_stats": {}
        }
        
        self.logger.info(f"Started new poker simulation session: {self.session_id}")
    
    def log_hand(
        self, 
        hand_num: int, 
        actions: List[Any], 
        winners: List[int], 
        pot: int, 
        players: List[Any],
        community_cards: List[Any] = None
    ) -> None:
        """
        Log information about a completed hand.
        
        Args:
            hand_num: The hand number
            actions: List of actions performed
            winners: List of player indices who won
            pot: Final pot size
            players: List of player objects
            community_cards: List of community cards
        """
        self.logger.info(f"Hand #{hand_num} complete - Pot: {pot}, Winners: {winners}")
        
        # Convert actions to serializable format
        action_data = []
        for action in actions:
            if hasattr(action, 'to_dict'):
                action_data.append(action.to_dict())
            else:
                action_data.append({
                    "type": str(action.get_type()),
                    "amount": action.get_amount() if hasattr(action, 'get_amount') else 0
                })
        
        # Record community cards
        card_data = []
        if community_cards:
            for card in community_cards:
                if hasattr(card, 'to_string'):
                    card_data.append(card.to_string())
                else:
                    card_data.append(str(card))
        
        # Collect hand data
        hand_data = {
            "hand_num": hand_num,
            "actions": action_data,
            "winners": winners,
            "pot": pot,
            "player_stacks": {},
            "community_cards": card_data
        }
        
        # Record player stacks and cards
        for i, player in enumerate(players):
            hand_data["player_stacks"][i] = player.get_stack()
            
            # Log hole cards if available
            if hasattr(player, 'get_hole_cards') and player.get_hole_cards():
                cards = player.get_hole_cards()
                hand_data[f"player_{i}_cards"] = [
                    card.to_string() if hasattr(card, 'to_string') else str(card)
                    for card in cards
                ]
            
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
    
    def log_blind_change(self, small_blind: int, big_blind: int, hand_num: int) -> None:
        """
        Log a change in blind levels.
        
        Args:
            small_blind: New small blind value
            big_blind: New big blind value
            hand_num: Hand number when change occurred
        """
        self.logger.info(f"Blinds changed at hand #{hand_num}: SB={small_blind}, BB={big_blind}")
        
        self.history["blinds_history"].append({
            "hand": hand_num,
            "small_blind": small_blind,
            "big_blind": big_blind
        })
    
    def log_tournament_end(
        self, 
        winner: int, 
        hands_played: int, 
        total_time: float,
        prize_pool: int,
        player_ranks: List[Tuple[int, int]] = None  # [(player_id, rank)]
    ) -> None:
        """
        Log the end of a tournament.
        
        Args:
            winner: Index of the winning player
            hands_played: Total hands played
            total_time: Total time in seconds
            prize_pool: Total prize pool
            player_ranks: List of (player_id, rank) tuples
        """
        self.logger.info(f"Tournament ended - Winner: Player {winner}, Hands: {hands_played}")
        
        self.history["tournament_stats"] = {
            "winner": winner,
            "hands_played": hands_played,
            "total_time": total_time,
            "prize_pool": prize_pool,
            "end_time": datetime.now().isoformat(),
            "player_ranks": player_ranks or []
        }
    
    def log_error(self, error_msg: str, context: Dict[str, Any] = None) -> None:
        """
        Log an error that occurred during simulation.
        
        Args:
            error_msg: Error message
            context: Additional context information
        """
        self.logger.error(f"Error: {error_msg}", extra=context or {})
    
    def log_action(self, player_idx: int, action: Any) -> None:
        """
        Log a single player action.
        
        Args:
            player_idx: Index of player performing action
            action: The action performed
        """
        action_type = action.get_type() if hasattr(action, 'get_type') else str(action)
        action_amount = action.get_amount() if hasattr(action, 'get_amount') else 0
        
        self.logger.debug(f"Player {player_idx} performs {action_type} {action_amount}")
    
    def export_data(self, filename: Optional[str] = None) -> str:
        """
        Export collected data to a file.
        
        Args:
            filename: Base filename (without extension)
            
        Returns:
            Path to the exported file
        """
        if filename is None:
            filename = f"poker_sim_{self.session_id}"
        
        filepath = os.path.join(self.log_dir, filename)
        
        if self.export_format == "json":
            with open(f"{filepath}.json", "w") as f:
                json.dump(self.history, f, indent=4)
            self.logger.info(f"Exported data to {filepath}.json")
            return f"{filepath}.json"
        
        elif self.export_format == "csv":
            # Export bankroll history
            bankroll_file = f"{filepath}_bankrolls.csv"
            with open(bankroll_file, "w", newline='') as f:
                writer = csv.writer(f)
                
                # Header row with player IDs
                writer.writerow(["Hand"] + [f"Player_{i}" for i in sorted(self.history["player_bankrolls"].keys())])
                
                # Data rows
                max_hands = max(len(rolls) for rolls in self.history["player_bankrolls"].values())
                for hand_idx in range(max_hands):
                    row = [hand_idx]
                    for player_id in sorted(self.history["player_bankrolls"].keys()):
                        if hand_idx < len(self.history["player_bankrolls"][player_id]):
                            row.append(self.history["player_bankrolls"][player_id][hand_idx])
                        else:
                            row.append(None)
                    writer.writerow(row)
            
            # Export pot sizes
            pot_file = f"{filepath}_pots.csv"
            with open(pot_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Hand", "Pot"])
                for i, hand in enumerate(self.history["hands"]):
                    writer.writerow([i, hand["pot"]])
            
            self.logger.info(f"Exported data to {bankroll_file} and {pot_file}")
            return bankroll_file
        
        else:
            self.logger.warning(f"Unsupported export format: {self.export_format}")
            return ""