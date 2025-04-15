# Poker Simulator API Reference

## Core C++ Components

### Enumerations

#### `Suit`
Card suits used in the poker game.
- `CLUBS`: Club suit
- `DIAMONDS`: Diamond suit
- `HEARTS`: Heart suit
- `SPADES`: Spade suit

#### `Rank`
Card ranks from lowest to highest.
- `TWO` through `ACE`: Standard poker card ranks

#### `HandType`
Types of poker hands in ascending order of value.
- `HIGH_CARD`: Highest card when no other hand is made
- `PAIR`: Two cards of the same rank
- `TWO_PAIR`: Two different pairs
- `THREE_OF_A_KIND`: Three cards of the same rank
- `STRAIGHT`: Five cards in sequence (Ace can be high or low)
- `FLUSH`: Five cards of the same suit
- `FULL_HOUSE`: Three of a kind plus a pair
- `FOUR_OF_A_KIND`: Four cards of the same rank
- `STRAIGHT_FLUSH`: Straight with all cards of the same suit
- `ROYAL_FLUSH`: A, K, Q, J, 10 of the same suit

#### `GameStage`
Stages of a poker hand.
- `PREFLOP`: Initial stage after hole cards are dealt
- `FLOP`: After the first three community cards are dealt
- `TURN`: After the fourth community card is dealt
- `RIVER`: After the fifth and final community card is dealt
- `SHOWDOWN`: Final stage where remaining players show their cards

#### `ActionType`
Types of player actions.
- `FOLD`: Give up the hand
- `CHECK`: Pass action without betting (when no bet is required)
- `CALL`: Match the current bet
- `BET`: Make the first bet in a round
- `RAISE`: Increase an existing bet
- `ALL_IN`: Bet all remaining chips

### Classes

#### `Card`
Represents a playing card.

**Constructors:**
- `Card()`: Create an invalid card
- `Card(Rank rank, Suit suit)`: Create a card with specific rank and suit
- `Card(int id)`: Create a card from a unique ID (0-51)
- `Card(const std::string& str)`: Create a card from a string representation (e.g., "AH" for Ace of Hearts)

**Methods:**
- `Rank getRank()`: Get the card's rank
- `Suit getSuit()`: Get the card's suit
- `int getId()`: Get the card's unique ID (0-51)
- `std::string toString()`: Get string representation of the card (e.g., "AH")
- `bool isValid()`: Check if the card is valid
- `static Card fromString(const std::string& str)`: Create a card from a string (e.g., "AH")

#### `Deck`
Represents a standard 52-card deck.

**Constructors:**
- `Deck()`: Create a standard deck in order
- `Deck(const std::vector<Card>& cards)`: Create a deck with specific cards

**Methods:**
- `void shuffle()`: Shuffle the deck using default RNG
- `void shuffle(std::mt19937& rng)`: Shuffle the deck with a specific RNG
- `Card dealCard()`: Deal the top card from the deck
- `void reset()`: Reset the deck to a full, ordered state
- `int cardsRemaining()`: Get the number of cards remaining in the deck
- `const std::vector<Card>& getCards()`: Get all cards in the deck
- `bool removeCard(const Card& card)`: Remove a specific card from the deck

#### `HandEvaluator`
Evaluates poker hands to determine their strength.

**Constructors:**
- `HandEvaluator()`: Create a hand evaluator

**Methods:**
- `int evaluate(const std::vector<Card>& cards)`: Evaluate a hand of 5-7 cards
- `int evaluate(const std::vector<Card>& holeCards, const std::vector<Card>& communityCards)`: Evaluate a hand from hole cards and community cards
- `HandType getHandType(int handValue)`: Get the hand type from a hand value
- `std::string getHandDescription(int handValue)`: Get a description of the hand (e.g., "Pair of Aces")
- `std::vector<Card> findBestHand(const std::vector<Card>& cards)`: Find the best 5-card hand from a set of cards

#### `Player`
Represents a poker player.

**Constructors:**
- `Player(int id, int64_t initialStack, const std::string& name = "")`: Create a player with an ID, initial stack, and optional name

**Methods:**
- `int getId()`: Get the player's ID
- `std::string getName()`: Get the player's name
- `void setHoleCards(const std::array<Card, NUM_HOLE_CARDS>& cards)`: Set the player's hole cards
- `const std::array<Card, NUM_HOLE_CARDS>& getHoleCards()`: Get the player's hole cards
- `int64_t getStack()`: Get the player's current chip stack
- `void adjustStack(int64_t amount)`: Add or remove chips from the player's stack
- `int64_t getCurrentBet()`: Get the player's current bet in this round
- `void setCurrentBet(int64_t amount)`: Set the player's current bet
- `void resetCurrentBet()`: Reset the player's current bet to zero
- `bool isAllIn()`: Check if the player is all-in
- `bool hasFolded()`: Check if the player has folded
- `void setFolded(bool folded)`: Set the player's folded status
- `bool isActive()`: Check if the player is active in the hand
- `void resetForNewHand()`: Reset the player's state for a new hand
- `std::string toString()`: Get a string representation of the player

#### `Action`
Represents a poker action.

**Constructors:**
- `Action()`: Create a default action (FOLD)
- `Action(ActionType type)`: Create an action with a specific type
- `Action(ActionType type, int64_t amount)`: Create an action with a type and bet amount

**Methods:**
- `ActionType getType()`: Get the action type
- `int64_t getAmount()`: Get the bet amount
- `static Action fold()`: Create a fold action
- `static Action check()`: Create a check action
- `static Action call()`: Create a call action
- `static Action bet(int64_t amount)`: Create a bet action
- `static Action raise(int64_t amount)`: Create a raise action
- `static Action allIn(int64_t amount)`: Create an all-in action
- `std::string toString()`: Get a string representation of the action

#### `GameState`
Represents the state of a poker game.

**Constructors:**
- `GameState(int numPlayers, int64_t initialStack, int64_t smallBlind, int64_t bigBlind)`: Create a game state

**Methods:**
- `void resetForNewHand()`: Reset the state for a new hand
- `void dealHoleCards()`: Deal hole cards to all players
- `void dealFlop()`: Deal the flop (first 3 community cards)
- `void dealTurn()`: Deal the turn (4th community card)
- `void dealRiver()`: Deal the river (5th community card)
- `void advanceStage()`: Advance to the next stage of the hand
- `GameStage getCurrentStage()`: Get the current stage of the hand
- `const std::vector<Card>& getCommunityCards()`: Get the community cards
- `const std::vector<Player>& getPlayers()`: Get all players
- `std::vector<Player>& getPlayersMutable()`: Get mutable reference to players
- `int64_t getPot()`: Get the current pot size
- `int getCurrentPlayerIndex()`: Get the index of the current player to act
- `int getDealerPosition()`: Get the dealer position
- `int64_t getSmallBlind()`: Get the small blind amount
- `int64_t getBigBlind()`: Get the big blind amount
- `int64_t getCurrentMinBet()`: Get the current minimum bet/raise
- `int64_t getMinRaise()`: Get the minimum raise amount
- `void applyAction(const Action& action)`: Apply a player action to the game state
- `std::vector<Action> getLegalActions()`: Get all legal actions for the current player
- `bool isHandOver()`: Check if the current hand is over
- `std::vector<int> getWinners()`: Get the indices of winners
- `int getHandValue(int playerIdx)`: Get a player's hand value
- `std::string getHandDescription(int playerIdx)`: Get a description of a player's hand
- `uint64_t getSeed()`: Get the current random seed
- `void setSeed(uint64_t seed)`: Set the random seed
- `const std::vector<Action>& getActionHistory()`: Get the history of all actions in the current hand
- `std::string toString()`: Get a string representation of the game state

#### `SpinGoGame`
Represents a Spin & Go poker tournament.

**Constructors:**
- `SpinGoGame(int numPlayers = 3, int64_t buyIn = 500, int64_t smallBlind = 10, int64_t bigBlind = 20, float prizeMultiplier = 2.0f)`: Create a Spin & Go game

**Methods:**
- `void play(int maxHands = -1)`: Play the game (up to maxHands if specified)
- `void playHand()`: Play a single hand
- `void playToCompletion()`: Play until the tournament is over
- `GameState* getGameState()`: Get the current game state
- `bool isTournamentOver()`: Check if the tournament is over
- `int getTournamentWinner()`: Get the index of the tournament winner
- `int64_t getPrizePool()`: Get the total prize pool
- `void setCallback(const std::function<void(const std::string&)>& callback)`: Set a callback for game events
- `void setSeed(uint64_t seed)`: Set the random seed
- `std::string toString()`: Get a string representation of the game
- `void addBlindLevel(int64_t smallBlind, int64_t bigBlind, int hands)`: Add a blind level that activates after a certain number of hands
- `std::vector<int> getLastHandWinners()`: Get the winners of the last hand played
- `int64_t getPlayerWinnings(int playerIdx)`: Get a player's total winnings from the tournament

## Python Extensions

### `poker.logging.GameLogger`

A class for logging poker game events and statistics.

**Constructor:**
- `GameLogger(log_dir="logs", log_level="INFO", export_format="json", session_id=None)`: Create a game logger

**Methods:**
- `log_hand(hand_num, actions, winners, pot, players, community_cards=None)`: Log a completed hand
- `log_blind_change(small_blind, big_blind, hand_num)`: Log a change in blind levels
- `log_tournament_end(winner, hands_played, total_time, prize_pool, player_ranks=None)`: Log the end of a tournament
- `log_error(error_msg, context=None)`: Log an error
- `log_action(player_idx, action)`: Log a player action
- `export_data(filename=None)`: Export collected data to a file

### `poker.visualization.PokerVisualizer`

A class for visualizing poker game data.

**Constructor:**
- `PokerVisualizer(data_source, output_dir="visualizations")`: Create a visualizer from data

**Methods:**
- `plot_bankrolls(save_path=None, show=True)`: Plot player bankrolls over time
- `plot_pot_sizes(save_path=None, show=True)`: Plot pot sizes across hands
- `plot_win_distribution(save_path=None, show=True)`: Plot distribution of wins by player
- `plot_blind_progression(save_path=None, show=True)`: Plot blind progression over time
- `plot_action_types(save_path=None, show=True)`: Plot distribution of action types
- `create_tournament_summary(save_path=None, show=True)`: Create a visual tournament summary
- `create_all_visualizations(show=False)`: Create all available visualizations

### `poker.testing`

Module for stress testing and tournament simulation.

**Functions:**
- `stress_test_hand_evaluation(num_iterations=100000)`: Stress test hand evaluation
- `run_tournament_simulation(...)`: Run a complete tournament simulation
- `parallel_tournament_simulations(num_tournaments=10, num_threads=4, **kwargs)`: Run multiple tournaments in parallel
- `test_hand_evaluator_accuracy()`: Test the accuracy of the hand evaluator

## (To Be Implemented) Advanced AI Algorithms

### `poker.deep_cfr.DeepCFRSolver`

Implementation of the Deep Counterfactual Regret Minimization algorithm.

**Constructor:**
- `DeepCFRSolver(game, policy_network_params, advantage_network_params, num_traversals=1000, num_iterations=100)`: Create a Deep CFR solver

**Methods:**
- `train(num_iterations)`: Train the solver for a number of iterations
- `solve()`: Generate an approximate Nash equilibrium
- `action_probabilities(game_state, player_idx)`: Get action probabilities for a player in a state
- `save_model(path)`: Save the trained model
- `load_model(path)`: Load a trained model

### `poker.rl_interface.RLEnvironment`

Reinforcement learning environment for poker.

**Constructor:**
- `RLEnvironment(num_players=3, initial_stack=1000, small_blind=10, big_blind=20)`: Create an RL environment

**Methods:**
- `reset()`: Reset the environment to a new episode
- `step(action)`: Take a step in the environment with an action
- `render()`: Render the current state
- `close()`: Close the environment
- `get_state_shape()`: Get the shape of state observations
- `get_action_space()`: Get the action space

## Usage Examples

### Basic Game Simulation
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

# Play a single hand
game.play_hand()

# Get the game state
state = game.get_game_state()

# Check if the tournament is over
if game.is_tournament_over():
    winner = game.get_tournament_winner()
    print(f"Tournament winner: Player {winner}")
    print(f"Prize pool: {game.get_prize_pool()}")
```

### Logging and Visualization
```python
from poker.logging import GameLogger
from poker.visualization import PokerVisualizer

# Create a logger
logger = GameLogger(log_dir="logs", export_format="json")

# Log a hand
logger.log_hand(
    hand_num=1,
    actions=state.get_action_history(),
    winners=[0, 1],  # Players 0 and 1 split the pot
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

### Stress Testing
```python
from poker.testing import stress_test_hand_evaluation

# Run a stress test of the hand evaluator
results = stress_test_hand_evaluation(num_iterations=1000000)
print(f"Hand evaluation speed: {results['hands_per_second']:.2f} hands/second")
```