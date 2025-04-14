#pragma once

#include "poker/action.h"
#include "poker/card.h"
#include "poker/constants.h"
#include "poker/deck.h"
#include "poker/hand_evaluator.h"
#include "poker/player.h"
#include <array>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace poker {

/**
 * Represents the state of a poker game.
 * Tracks players, community cards, pot, and game progress.
 */
class GameState {
public:
    // Create a game state with a set number of players and initial stacks
    GameState(int num_players, int64_t initial_stack, int64_t small_blind, int64_t big_blind);
    
    // Reset the game state for a new hand
    void resetForNewHand();
    
    // Deal hole cards to all players
    void dealHoleCards();
    
    // Deal the flop (first 3 community cards)
    void dealFlop();
    
    // Deal the turn (4th community card)
    void dealTurn();
    
    // Deal the river (5th community card)
    void dealRiver();
    
    // Advance to the next stage of the game
    void advanceStage();
    
    // Get the current stage of the game
    GameStage getCurrentStage() const;
    
    // Get the community cards
    const std::vector<Card>& getCommunityCards() const;
    
    // Get the players
    const std::vector<Player>& getPlayers() const;
    
    // Get a mutable reference to the players (for RL algorithms)
    std::vector<Player>& getPlayersMutable();
    
    // Get the total pot size
    int64_t getPot() const;
    
    // Get the current player's ID (whose turn it is)
    int getCurrentPlayerIndex() const;
    
    // Get the dealer button position
    int getDealerPosition() const;
    
    // Get the small blind amount
    int64_t getSmallBlind() const;
    
    // Get the big blind amount
    int64_t getBigBlind() const;
    
    // Get the current minimum bet to stay in the hand
    int64_t getCurrentMinBet() const;
    
    // Get the minimum raise amount
    int64_t getMinRaise() const;
    
    // Process an action from the current player
    void applyAction(const Action& action);
    
    // Get a list of legal actions for the current player
    std::vector<Action> getLegalActions() const;
    
    // Check if the hand is over
    bool isHandOver() const;
    
    // Get the winner(s) of the current hand
    std::vector<int> getWinners() const;
    
    // Get the hand value for a player
    uint32_t getHandValue(int player_index) const;
    
    // Get the hand description for a player
    std::string getHandDescription(int player_index) const;
    
    // Get the seed for the random number generator
    uint64_t getSeed() const;
    
    // Set the seed for the random number generator
    void setSeed(uint64_t seed);
    
    // Get a string representation of the game state
    std::string toString() const;

private:
    std::vector<Player> players_;
    std::vector<Card> community_cards_;
    Deck deck_;
    HandEvaluator hand_evaluator_;
    
    int dealer_position_ = 0;
    int current_player_index_ = 0;
    int last_aggressor_ = -1;
    
    int64_t small_blind_;
    int64_t big_blind_;
    int64_t current_min_bet_ = 0;
    int64_t min_raise_ = 0;
    int64_t pot_ = 0;
    
    GameStage current_stage_ = GameStage::PREFLOP;
    
    std::mt19937 rng_;
    uint64_t seed_;
    
    std::vector<Action> action_history_;
    
    // Move to the next player
    void advanceToNextPlayer();
    
    // Post blinds at the start of a hand
    void postBlinds();
    
    // Distribute the pot to the winner(s)
    void distributePot();
    
    // Check if the current betting round is over
    bool isBettingRoundOver() const;
    
    // Get number of active players
    int getActivePlayerCount() const;
    
    // Get number of players who haven't folded or are all-in
    int getPlayersInHand() const;
    
    // Move all current bets to the pot
    void moveBetsToPot();
};

} // namespace poker
