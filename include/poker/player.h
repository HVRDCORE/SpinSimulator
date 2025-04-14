#pragma once

#include "poker/card.h"
#include <array>
#include <string>
#include <vector>

namespace poker {

/**
 * Represents a player in a poker game.
 */
class Player {
public:
    // Create a player with an initial stack
    Player(int id, int64_t initial_stack, const std::string& name = "");
    
    // Get the player's ID
    int getId() const;
    
    // Get the player's name
    const std::string& getName() const;
    
    // Set the player's hole cards
    void setHoleCards(const std::array<Card, NUM_HOLE_CARDS>& cards);
    
    // Get the player's hole cards
    const std::array<Card, NUM_HOLE_CARDS>& getHoleCards() const;
    
    // Get the player's current stack
    int64_t getStack() const;
    
    // Modify the player's stack (positive for add, negative for remove)
    void adjustStack(int64_t amount);
    
    // Get the amount the player has bet in the current round
    int64_t getCurrentBet() const;
    
    // Set the amount the player has bet in the current round
    void setCurrentBet(int64_t amount);
    
    // Reset the player's current bet (e.g., between betting rounds)
    void resetCurrentBet();
    
    // Check if the player is all-in
    bool isAllIn() const;
    
    // Check if the player has folded
    bool hasFolded() const;
    
    // Set the player's folded status
    void setFolded(bool folded);
    
    // Check if the player is active (has cards and hasn't folded)
    bool isActive() const;
    
    // Reset the player for a new hand
    void resetForNewHand();
    
    // String representation
    std::string toString() const;

private:
    int id_;
    std::string name_;
    std::array<Card, NUM_HOLE_CARDS> hole_cards_;
    int64_t stack_;
    int64_t current_bet_ = 0;
    bool has_folded_ = false;
    bool has_cards_ = false;
};

} // namespace poker
