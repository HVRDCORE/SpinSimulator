# Poker Simulator Documentation

## Overview

This project is a high-performance C++ Spin&Go poker simulator with Python bindings, designed for reinforcement learning and tree traversal algorithms in poker strategy optimization. The core simulation engine is implemented in C++ for maximum performance, while Python bindings provide easy integration with ML frameworks.

## Architecture

The project is structured into the following components:

1. **C++ Core Library** - High-performance game logic and simulation engine
2. **Python Bindings** - Interface between C++ core and Python
3. **Poker AI Algorithms** - Deep CFR and MCCFR implementations
4. **Visualization & Logging** - Tools for data collection and visualization
5. **Testing Utilities** - Comprehensive testing suite

## Installation

### Prerequisites

- C++ compiler with C++14 support
- CMake (3.10+)
- Python 3.8+
- pybind11
- numpy, tensorflow (for RL components)
- matplotlib, pandas, seaborn (for visualization)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/poker-simulator.git
cd poker-simulator

# Install dependencies
pip install -r requirements.txt

# Build the C++ library
python build_poker_core.py
```

## Core Components

### C++ Components

#### Cards & Deck (`card.cpp`, `deck.cpp`)

The fundamental components representing playing cards and a shufflable deck.

```cpp
// Card representation
Card card = Card(HEARTS, KING);
std::string card_str = card.toString();  // "KH"

// Deck operations
Deck deck;
deck.shuffle();
Card dealt_card = deck.dealCard();
```

#### Hand Evaluator (`hand_evaluator.cpp`)

Fast poker hand evaluation using optimized algorithms.

```cpp
HandEvaluator evaluator;
std::vector<Card> cards = /* 7 cards (5 community + 2 hole) */;
int hand_value = evaluator.evaluate(cards);
HandType type = evaluator.getHandType(hand_value);
```

#### Game State (`game_state.cpp`)

Manages the current state of a poker hand, including players, pot, community cards, and legal actions.

```cpp
GameState state(3, 1000, 10, 20);  // 3 players, 1000 chips, SB=10, BB=20
state.dealHoleCards();
state.dealFlop();
std::vector<Action> legal_actions = state.getLegalActions();
```

#### Spin&Go Game (`spingo_game.cpp`)

Implementation of a Spin&Go tournament format.

```cpp
SpinGoGame game(3, 1000, 10, 20, 2.0);  // 3 players, 1000 buy-in, SB=10, BB=20, 2x multiplier
game.setSeed(42);
game.playHand();
bool is_over = game.isTournamentOver();
```

### Python Components

#### Core Module (`poker_core`)

```python
import poker_core as pc

# Create a game
game = pc.SpinGoGame(
    num_players=3,
    buy_in=1000,
    small_blind=10,
    big_blind=20
)

# Access game state
state = game.get_game_state()
```

#### Logging (`poker.logging`)

Comprehensive logging system for tracking game progress and statistics.

```python
from poker.logging import GameLogger

logger = GameLogger(log_dir="logs", export_format="json")
logger.log_hand(hand_num, actions, winners, pot, players, community_cards)
logger.export_data()
```

#### Visualization (`poker.visualization`)

Tools for visualizing game data and statistics.

```python
from poker.visualization import PokerVisualizer

visualizer = PokerVisualizer(data_file, output_dir="visualizations")
visualizer.plot_bankrolls()
visualizer.plot_pot_sizes()
visualizer.create_all_visualizations()
```

#### Testing (`poker.testing`)

Utilities for stress testing and tournament simulation.

```python
from poker.testing import run_tournament_simulation, stress_test_hand_evaluation

# Run stress test
results = stress_test_hand_evaluation(num_iterations=100000)

# Simulate tournament
tournament_results = run_tournament_simulation(
    num_players=3,
    max_hands=100,
    visualize=True
)
```

## Usage Examples

### Simple Game Simulation

```python
import poker_core as pc

# Create a game
game = pc.SpinGoGame(
    num_players=3,
    buy_in=1000,
    small_blind=10,
    big_blind=20
)

# Set seed for reproducibility
game.set_seed(42)

# Get game state
state = game.get_game_state()

# Deal hole cards
state.deal_hole_cards()

# Play a hand
while not state.is_hand_over():
    current_player_idx = state.get_current_player_index()
    legal_actions = state.get_legal_actions()
    
    # Choose an action (e.g., from an AI agent)
    action = choose_action(legal_actions)
    
    # Apply the action
    state.apply_action(action)

# Check winners
winners = game.get_last_hand_winners()
```

### Tournament with Visualization

```python
from poker.logging import GameLogger
from poker.visualization import PokerVisualizer

# Create logger
logger = GameLogger(log_dir="logs", export_format="json")

# Create game
game = pc.SpinGoGame(num_players=3, buy_in=1000, small_blind=10, big_blind=20)

# Play tournament
for _ in range(20):  # Play 20 hands
    # Play a hand
    game.play_hand()
    
    # Log results
    state = game.get_game_state()
    logger.log_hand(
        hand_num=i+1,
        actions=state.get_action_history(),
        winners=game.get_last_hand_winners(),
        pot=state.get_pot(),
        players=state.get_players(),
        community_cards=state.get_community_cards()
    )

# Export data
data_file = logger.export_data()

# Create visualizations
visualizer = PokerVisualizer(data_file, output_dir="visualizations")
visualizer.create_all_visualizations()
```

### Reinforcement Learning Integration

```python
from poker.rl_interface import RLEnvironment

# Create RL environment
env = RLEnvironment(
    num_players=3,
    initial_stack=1000,
    small_blind=10,
    big_blind=20
)

# RL training loop
obs = env.reset()
for _ in range(1000):
    action = model.predict(obs)  # Your RL model
    next_obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
    else:
        obs = next_obs
```

## Performance Benchmarks

The simulator achieves the following performance metrics:

- **Hand Evaluation**: ~61,000 hands/second
- **Game Simulation**: ~1,200 hands/second
- **Python Bindings**: ~9,200 iterations/second

These benchmarks make the simulator suitable for training advanced AI algorithms that require millions of game simulations.

## Advanced Topics

### Multi-threading Support

The simulator supports parallel tournament simulations for improved performance:

```python
from poker.testing import parallel_tournament_simulations

results = parallel_tournament_simulations(
    num_tournaments=10,
    num_threads=4,
    num_players=3,
    max_hands=100
)
```

### Customizing Blind Schedules

Spin&Go tournaments can be configured with custom blind schedules:

```python
game = pc.SpinGoGame(num_players=3, buy_in=1000, small_blind=10, big_blind=20)

# Add blind levels
game.add_blind_level(10, 20, 0)    # Initial level
game.add_blind_level(20, 40, 10)   # After 10 hands
game.add_blind_level(30, 60, 20)   # After 20 hands
```

## Extending the Simulator

### Implementing Custom Agents

You can implement custom poker agents by creating functions that select actions:

```python
def my_poker_agent(legal_actions, player_idx, state):
    """
    Custom poker agent implementation.
    
    Args:
        legal_actions: List of legal Action objects
        player_idx: Player index
        state: Current game state
        
    Returns:
        Selected Action object
    """
    # Your strategy logic here
    return chosen_action
```

### Adding New Game Formats

The simulator can be extended to support other poker variants by implementing new game classes in C++.

## Troubleshooting

### Common Issues

1. **Installation Errors**: Make sure all dependencies are installed and your C++ compiler supports C++14.
2. **Build Failures**: Delete CMakeCache.txt and rebuild if you encounter build issues.
3. **Performance Problems**: For optimal performance, compile with release mode (`-DCMAKE_BUILD_TYPE=Release`).

## License

This project is licensed under the MIT License - see the LICENSE file for details.