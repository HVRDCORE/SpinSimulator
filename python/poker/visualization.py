#!/usr/bin/env python3
"""
Visualization tools for poker simulations.
This module provides various ways to visualize poker game data,
including player bankrolls, pot sizes, win rates, and more.
"""
import os
import json
import csv
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class PokerVisualizer:
    """
    Visualizer for poker simulation data.
    Generates various plots and charts to analyze poker game results.
    """
    
    def __init__(
        self, 
        data_source: Union[str, Dict[str, Any]], 
        output_dir: str = "visualizations"
    ):
        """
        Initialize the visualizer.
        
        Args:
            data_source: Either a path to a JSON file or a dictionary with simulation data
            output_dir: Directory where to save visualizations
        """
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
    
    def plot_bankrolls(self, save_path: Optional[str] = None, show: bool = True) -> str:
        """
        Plot the evolution of each player's bankroll over time.
        
        Args:
            save_path: Path to save the plot (if None, auto-generated)
            show: Whether to display the plot
            
        Returns:
            Path to the saved plot file
        """
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
    
    def plot_pot_sizes(self, save_path: Optional[str] = None, show: bool = True) -> str:
        """
        Plot the evolution of pot sizes across hands.
        
        Args:
            save_path: Path to save the plot (if None, auto-generated)
            show: Whether to display the plot
            
        Returns:
            Path to the saved plot file
        """
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
    
    def plot_win_distribution(self, save_path: Optional[str] = None, show: bool = True) -> str:
        """
        Plot the distribution of wins by player.
        
        Args:
            save_path: Path to save the plot (if None, auto-generated)
            show: Whether to display the plot
            
        Returns:
            Path to the saved plot file
        """
        if not save_path:
            save_path = os.path.join(self.output_dir, "win_distribution.png")
        
        plt.figure(figsize=(12, 8))
        
        # Count wins per player
        player_wins = {}
        for winner_data in self.data["winners"]:
            player_id = winner_data["player"]
            if player_id not in player_wins:
                player_wins[player_id] = 0
            player_wins[player_id] += 1
        
        # Create bar chart
        player_ids = list(player_wins.keys())
        win_counts = list(player_wins.values())
        
        plt.bar(player_ids, win_counts, color=self.colors[:len(player_ids)])
        
        plt.title("Win Distribution by Player", fontsize=16)
        plt.xlabel("Player ID", fontsize=14)
        plt.ylabel("Number of Wins", fontsize=14)
        plt.xticks(player_ids)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add win count labels above bars
        for i, v in enumerate(win_counts):
            plt.text(player_ids[i], v + 0.1, str(v), ha='center')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_blind_progression(self, save_path: Optional[str] = None, show: bool = True) -> str:
        """
        Plot the progression of blinds over the course of the tournament.
        
        Args:
            save_path: Path to save the plot (if None, auto-generated)
            show: Whether to display the plot
            
        Returns:
            Path to the saved plot file or empty string if no blind history
        """
        if not self.data.get("blinds_history"):
            return ""
            
        if not save_path:
            save_path = os.path.join(self.output_dir, "blind_progression.png")
        
        plt.figure(figsize=(12, 8))
        
        # Extract blind history
        hand_numbers = [entry["hand"] for entry in self.data["blinds_history"]]
        small_blinds = [entry["small_blind"] for entry in self.data["blinds_history"]]
        big_blinds = [entry["big_blind"] for entry in self.data["blinds_history"]]
        
        # Start from hand 1
        hand_numbers = [1] + hand_numbers
        small_blinds = [small_blinds[0]] + small_blinds
        big_blinds = [big_blinds[0]] + big_blinds
        
        plt.plot(hand_numbers, small_blinds, marker='o', linestyle='-', label="Small Blind", color=self.colors[0])
        plt.plot(hand_numbers, big_blinds, marker='s', linestyle='-', label="Big Blind", color=self.colors[1])
        
        plt.title("Blind Progression", fontsize=16)
        plt.xlabel("Hand Number", fontsize=14)
        plt.ylabel("Blind Size", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_action_types(self, save_path: Optional[str] = None, show: bool = True) -> str:
        """
        Plot the distribution of action types across all hands.
        
        Args:
            save_path: Path to save the plot (if None, auto-generated)
            show: Whether to display the plot
            
        Returns:
            Path to the saved plot file
        """
        if not save_path:
            save_path = os.path.join(self.output_dir, "action_types.png")
        
        plt.figure(figsize=(12, 8))
        
        # Count action types
        action_counts = {}
        for hand in self.data["hands"]:
            for action in hand["actions"]:
                action_type = action["type"]
                if action_type not in action_counts:
                    action_counts[action_type] = 0
                action_counts[action_type] += 1
        
        # Create pie chart
        labels = list(action_counts.keys())
        sizes = list(action_counts.values())
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=self.colors[:len(labels)])
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.title("Distribution of Action Types", fontsize=16)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def create_tournament_summary(self, save_path: Optional[str] = None, show: bool = True) -> str:
        """
        Create a visual summary of tournament statistics.
        
        Args:
            save_path: Path to save the plot (if None, auto-generated)
            show: Whether to display the plot
            
        Returns:
            Path to the saved plot file
        """
        if not self.data.get("tournament_stats"):
            return ""
            
        if not save_path:
            save_path = os.path.join(self.output_dir, "tournament_summary.png")
        
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Bankroll final values (top left)
        player_ids = []
        final_stacks = []
        
        for player_id, bankroll in self.data["player_bankrolls"].items():
            player_ids.append(int(player_id))
            final_stacks.append(bankroll[-1])
        
        axs[0, 0].bar(player_ids, final_stacks, color=self.colors[:len(player_ids)])
        axs[0, 0].set_title("Final Player Stacks", fontsize=14)
        axs[0, 0].set_xlabel("Player ID", fontsize=12)
        axs[0, 0].set_ylabel("Stack Size", fontsize=12)
        axs[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Hands played stats (top right)
        stats = self.data["tournament_stats"]
        hands_played = stats["hands_played"]
        total_time = stats["total_time"]
        hands_per_second = hands_played / total_time if total_time > 0 else 0
        
        # Create a small table with tournament stats
        table_data = [
            ["Hands Played", hands_played],
            ["Total Time (s)", f"{total_time:.2f}"],
            ["Hands/Second", f"{hands_per_second:.2f}"],
            ["Prize Pool", stats["prize_pool"]],
            ["Winner", f"Player {stats['winner']}"]
        ]
        
        axs[0, 1].axis('off')
        table = axs[0, 1].table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        axs[0, 1].set_title("Tournament Statistics", fontsize=14)
        
        # 3. Win distribution (bottom left)
        player_wins = {}
        for winner_data in self.data["winners"]:
            player_id = winner_data["player"]
            if player_id not in player_wins:
                player_wins[player_id] = 0
            player_wins[player_id] += 1
        
        win_percentages = {}
        total_hands = len(self.data["hands"])
        for player_id, wins in player_wins.items():
            win_percentages[player_id] = (wins / total_hands) * 100
        
        axs[1, 0].bar(list(win_percentages.keys()), list(win_percentages.values()), 
                     color=self.colors[:len(win_percentages)])
        axs[1, 0].set_title("Win Percentage by Player", fontsize=14)
        axs[1, 0].set_xlabel("Player ID", fontsize=12)
        axs[1, 0].set_ylabel("Win Percentage (%)", fontsize=12)
        axs[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Pot size distribution (bottom right)
        pot_sizes = [hand["pot"] for hand in self.data["hands"]]
        
        axs[1, 1].hist(pot_sizes, bins=20, color=self.colors[0], alpha=0.7)
        axs[1, 1].set_title("Pot Size Distribution", fontsize=14)
        axs[1, 1].set_xlabel("Pot Size", fontsize=12)
        axs[1, 1].set_ylabel("Frequency", fontsize=12)
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def create_all_visualizations(self, show: bool = False) -> List[str]:
        """
        Create all available visualizations.
        
        Args:
            show: Whether to display the plots
            
        Returns:
            List of paths to all created visualization files
        """
        paths = []
        
        # Create all plots
        paths.append(self.plot_bankrolls(show=show))
        paths.append(self.plot_pot_sizes(show=show))
        paths.append(self.plot_win_distribution(show=show))
        
        blind_path = self.plot_blind_progression(show=show)
        if blind_path:
            paths.append(blind_path)
            
        paths.append(self.plot_action_types(show=show))
        
        summary_path = self.create_tournament_summary(show=show)
        if summary_path:
            paths.append(summary_path)
        
        return paths