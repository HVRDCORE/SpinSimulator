# Spin&Go Poker Simulator

A high-performance C++ Spin&Go poker simulator with Python bindings for reinforcement learning and tree traversal algorithms. Designed for AI research and poker strategy optimization.

## Features

- **High Performance**: Fast C++ core implementation of Texas Hold'em rules for Spin&Go tournaments
- **Python Integration**: Seamless Python bindings via pybind11 for integration with ML frameworks
- **Reinforcement Learning**: Support for RL environments compatible with OpenAI Gym and Ray/RLlib
- **Tree Traversal Algorithms**: Implementation of MCCFR and Deep CFR for solving poker games
- **Optimized Hand Evaluation**: Efficient algorithms for poker hand evaluation
- **Visualization & Logging**: Comprehensive data collection and visualization tools
- **Comprehensive Testing**: Includes unit tests, benchmarks, and example scripts

## Benchmarks

The simulator achieves excellent performance:

- Hand evaluation: ~10,000-25,000 hands/second
- Game simulation: ~1,000-2,000 complete games/second
- Python bindings: ~5,000-7,000 operations/second

## Requirements

- C++17 compatible compiler (GCC 8+, Clang 6+, or MSVC 2019+)
- CMake 3.10+
- Python 3.8+
- pybind11 (automatically fetched by CMake if not found)
- NumPy
- TensorFlow 2.x (for Deep CFR)
- Ray/RLlib (for RL examples)

## Installation

### Direct Build

```bash
# Run the build script (fetches dependencies if needed)
python build_poker_core.py

# Verify the build with benchmarks
python python/examples/benchmark.py --benchmark all
```

### Using as a Python Package

```bash
# Install in development mode
pip install -e .

# Or install directly
pip install .
```

## Project Structure

- `include/poker/` - C++ header files
- `src/` - C++ implementation files
- `python/` - Python module and bindings
  - `python/poker/` - Python library files
  - `python/examples/` - Example scripts and benchmarks
- `tests/` - C++ unit tests

## Core Components

- **Card, Deck**: Basic card game elements
- **HandEvaluator**: Fast poker hand evaluation
- **Player**: Player state management
- **Action**: Poker actions (fold, check, call, bet, raise)
- **GameState**: Core game state machine
- **SpinGoGame**: Spin&Go tournament logic

## Reinforcement Learning Interface

The RL interface provides:

- Observation space encoding (one-hot cards, player stacks, etc.)
- Action space management (legal actions masking)
- Reward calculation based on money won/lost
- Episode termination conditions

## Example Usage

```python
import poker_core as pc

# Create a SpinGoGame
game = pc.SpinGoGame(
    num_players=3,
    buy_in=500,
    small_blind=10,
    big_blind=20,
    prize_multiplier=2.0
)

# Set a seed for reproducibility
game.set_seed(42)

# Get the game state
state = game.get_game_state()

# Deal cards and start playing
state.deal_hole_cards()

# Get legal actions for current player
legal_actions = state.get_legal_actions()

# Make a move
action = legal_actions[0]  # Choose first legal action
state.apply_action(action)

# Check if the hand is over
if state.is_hand_over():
    winners = state.get_winners()
    print(f"Winners: {winners}")
```

## Advanced Examples

See the `python/examples/` directory for more examples:

- `benchmark.py`: Performance benchmarks
- `simple_game_test.py`: Basic game simulation
- `tf_dqn_example.py`: TensorFlow Deep Q-Network implementation
- `simple_text_visualization.py`: Data visualization and logging (no dependencies)
- `visualization_demo.py`: Advanced visualization with matplotlib/seaborn

## Documentation

Comprehensive documentation is available in the project:

- `API.md`: Complete API reference for all classes and functions
- `DOCUMENTATION.md`: Detailed user guide with advanced examples
