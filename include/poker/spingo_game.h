#pragma once

#include "poker/game_state.h"
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace poker {

/**
 * Represents a complete Spin & Go poker game.
 * Handles multiple hands until the tournament is over.
 */
class SpinGoGame {
public:
    // Callback type for notifications about game events
    using GameCallback = std::function<void(const GameState&, const std::string&)>;
    
    // Create a new Spin & Go game with the given parameters
    SpinGoGame(int num_players = 3, 
              int64_t buy_in = 500,  // 500 chip buy-in
              int64_t small_blind = 10,
              int64_t big_blind = 20,
              float prize_multiplier = 2.0f);
    
    // Start the game and play until completion
    void play();
    
    // Play a single hand
    void playHand();
    
    // Play until the tournament is complete
    void playToCompletion();
    
    // Get the current game state
    const GameState& getGameState() const;
    
    // Get a mutable reference to the game state (for RL algorithms)
    GameState& getGameStateMutable();
    
    // Check if the tournament is over
    bool isTournamentOver() const;
    
    // Get the winner of the tournament (valid only if tournament is over)
    int getTournamentWinner() const;
    
    // Get the prize pool
    int64_t getPrizePool() const;
    
    // Set a callback for game events
    void setCallback(const GameCallback& callback);
    
    // Set the random seed
    void setSeed(uint64_t seed);
    
    // Get a string representation of the game
    std::string toString() const;

private:
    std::unique_ptr<GameState> game_state_;
    int64_t buy_in_;
    float prize_multiplier_;
    int64_t prize_pool_;
    std::mt19937 rng_;
    GameCallback callback_;
    
    // Blind schedule for increasing blinds over time
    struct BlindLevel {
        int hands;
        int64_t small_blind;
        int64_t big_blind;
    };
    
    std::vector<BlindLevel> blind_schedule_;
    int current_blind_level_ = 0;
    int hands_played_ = 0;
    
    // Update blinds if needed
    void updateBlinds();
    
    // Notify callback if set
    void notify(const std::string& message) const;
};

} // namespace poker
