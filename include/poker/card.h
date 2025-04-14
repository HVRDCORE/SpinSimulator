#pragma once

#include "poker/constants.h"
#include <iostream>
#include <string>

namespace poker {

/**
 * Represents a playing card with a rank and suit.
 * Uses a compact representation for efficiency.
 */
class Card {
public:
    // Default constructor creates an invalid card
    Card();
    
    // Create a card with specified rank and suit
    Card(Rank rank, Suit suit);
    
    // Create a card from an integer index (0-51)
    explicit Card(int id);
    
    // Create a card from a string (e.g., "Ah", "Ts", "2c")
    explicit Card(const std::string& card_str);
    
    // Get the rank of the card
    Rank getRank() const;
    
    // Get the suit of the card
    Suit getSuit() const;
    
    // Get the card ID (0-51)
    int getId() const;
    
    // Get a string representation of the card (e.g., "Ah")
    std::string toString() const;
    
    // Comparison operators
    bool operator==(const Card& other) const;
    bool operator!=(const Card& other) const;
    
    // Check if the card is valid
    bool isValid() const;

private:
    // Compact representation of the card: 0-51
    // Suit = id / 13, Rank = id % 13
    int id_;
};

// Stream output operator for Card
std::ostream& operator<<(std::ostream& os, const Card& card);

} // namespace poker
