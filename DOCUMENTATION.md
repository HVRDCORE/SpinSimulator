# Poker Simulator Documentation

## Overview

This project provides a high-performance poker simulator focused on Spin & Go tournaments, with special emphasis on reinforcement learning applications. The core simulation engine is written in C++ for efficiency, while providing Python bindings for easy integration with machine learning frameworks like TensorFlow and PyTorch.

## Architecture

The simulator follows a modular design with these primary components:

1. **Core C++ Engine**
   - Card representation and manipulation
   - Hand evaluation
   - Game state management
   - Tournament logic

2. **Python Bindings** 
   - Interface with the C++ core
   - Reinforcement learning environments
   - Visualization tools
   - Logging and data export

3. **AI Components**
   - Deep Q-Network (DQN) implementation
   - Deep Counterfactual Regret Minimization (CFRD)
   - Interface for custom agents

## Installation

### Requirements
- C++ compiler with C++17 support
- CMake (3.15+)
- Python 3.8+ with pip
- Libraries: TensorFlow, NumPy, Matplotlib, Seaborn

### Building from Source

```bash
# Clone the repository
git clone [repository-url]
cd poker-simulator

# Build the C++ core and Python bindings
python build_poker_core.py

# Run tests
python -m unittest discover tests
```

## Using the Simulator

### Basic Usage

```python
import poker_core as pc

# Create a Spin & Go game
game = pc.SpinGoGame(
    num_players=3,
    buy_in=1000,
    small_blind=10,
    big_blind=20,
    prize_multiplier=2.0
)

# Set a seed for reproducibility
game.set_seed(42)

# Play a hand
game.play_hand()

# Get the game state
state = game.get_game_state()

# Print information about the current state
print(f"Pot: {state.get_pot()}")
print(f"Current player: {state.get_current_player_index()}")
print(f"Hand is over: {state.is_hand_over()}")

# Continue playing until the tournament is over
while not game.is_tournament_over():
    game.play_hand()

# Get the winner
winner = game.get_tournament_winner()
print(f"Tournament winner: Player {winner}")
```

### Reinforcement Learning

The simulator provides a reinforcement learning environment that follows the OpenAI Gym interface:

```python
from poker.rl_interface import RLEnvironment

# Create the environment
env = RLEnvironment(num_players=3, initial_stack=1000, small_blind=10, big_blind=20)

# Reset the environment
observation = env.reset()

# Take steps in the environment
action = 1  # For example, CHECK action
next_observation, reward, done, info = env.step(action)
```

### Deep Q-Network Example

```python
from poker.rl_interface import RLEnvironment
from python.examples.tf_dqn_example import DQNAgent, train_dqn_agent

# Create environment
env = RLEnvironment(num_players=3, initial_stack=1000, small_blind=10, big_blind=20)

# Create agent
agent = DQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
    memory_size=100000,
    batch_size=64,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    learning_rate=0.001
)

# Train agent
episode_rewards = train_dqn_agent(
    env=env,
    agent=agent,
    episodes=1000,
    target_update_freq=100,
    batch_size=32,
    max_steps=200,
    render_freq=100,
    save_freq=100,
    save_dir="models/dqn"
)
```

### Deep CFR Example

```python
import poker_core as pc
from poker.deep_cfr import DeepCFRSolver

# Create a game
game = pc.SpinGoGame(num_players=3, buy_in=1000, small_blind=10, big_blind=20)

# Create solver
solver = DeepCFRSolver(
    game,
    hidden_layers=[256, 256],
    learning_rate=0.001,
    memory_size=1000000
)

# Train the solver
solver.train(
    iterations=50,
    traversals_per_iter=100,
    advantage_training_epochs=5,
    strategy_training_epochs=5,
    batch_size=128
)

# Use the trained strategy
action_probs = solver.get_strategy(game.get_game_state(), 0)
```

### Data Logging and Visualization

The simulator includes tools for logging game data and creating visualizations:

```python
from poker.logging import GameLogger
from poker.visualization import PokerVisualizer

# Create a logger
logger = GameLogger(log_dir="logs", export_format="json")

# Log a hand
logger.log_hand(
    hand_num=1,
    actions=state.get_action_history(),
    winners=[0],
    pot=100,
    players=state.get_players(),
    community_cards=state.get_community_cards()
)

# Export the data
data_file = logger.export_data()

# Create visualizations
visualizer = PokerVisualizer(data_file, output_dir="visualizations")
visualizer.create_all_visualizations()
```

## Advanced Usage

### Custom Agents

You can create custom agents to play poker by implementing a function that takes legal actions and returns one of them:

```python
def custom_agent(legal_actions, player_idx, state):
    """
    A custom poker agent.
    
    Args:
        legal_actions: List of legal Action objects
        player_idx: The player's index
        state: The current game state
        
    Returns:
        A poker Action object
    """
    # Always check if possible
    action_types = [a.get_type() for a in legal_actions]
    if pc.ActionType.CHECK in action_types:
        return next(a for a in legal_actions if a.get_type() == pc.ActionType.CHECK)
    
    # Otherwise, fold
    return next(a for a in legal_actions if a.get_type() == pc.ActionType.FOLD)
```

### Tournament Customization

You can customize tournaments with blind structures and prize multipliers:

```python
# Create game with customizable prize multiplier
game = pc.SpinGoGame(num_players=3, buy_in=1000, prize_multiplier=5.0)

# Add a blind structure
game.add_blind_level(10, 20, 0)    # Initial blinds
game.add_blind_level(20, 40, 5)    # After 5 hands
game.add_blind_level(50, 100, 10)  # After 10 hands
game.add_blind_level(100, 200, 15) # After 15 hands
```

### Parallel Simulations

The framework supports running multiple simulations in parallel:

```python
from poker.testing import parallel_tournament_simulations

# Run 10 tournaments in parallel with 4 threads
results = parallel_tournament_simulations(
    num_tournaments=10,
    num_threads=4,
    num_players=3,
    buy_in=1000,
    small_blind=10,
    big_blind=20
)

# Analyze results
win_counts = {i: 0 for i in range(3)}
for result in results:
    win_counts[result["winner"]] += 1

print(f"Win distribution: {win_counts}")
```

## Performance Benchmarks

The simulator is designed for high performance. Benchmark results on reference hardware:

- Hand evaluation: ~5-10 million hands per second
- Complete Spin & Go tournament: ~2,000-5,000 tournaments per second
- DQN training: ~500-1,000 steps per second

## Extending the Framework

The framework is designed to be extensible. Key extension points include:

1. **Custom card evaluation**: Extend the `HandEvaluator` class
2. **New game variants**: Create classes similar to `SpinGoGame`
3. **Agent algorithms**: Implement in the Python layer
4. **Reward functions**: Customize in the `RLInterface` class

## Troubleshooting

Common issues and solutions:

1. **Python can't find the module**: Ensure the `.so` file is in your Python path.
2. **Compilation errors**: Verify you have the correct C++ compiler and CMake version.
3. **TensorFlow errors**: TensorFlow 2.x is required for the machine learning components.
4. **Memory issues**: For large simulations, you may need to reduce batch sizes or training parameters.

## License

This project is licensed under the MIT License. See the LICENSE file for details.