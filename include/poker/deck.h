#pragma once

#include "poker/card.h"
#include <random>
#include <vector>

namespace poker {

/**
 * Represents a deck of playing cards.
 * Provides methods for shuffling and dealing cards.
 */
class Deck {
public:
    // Initialize a standard 52-card deck
    Deck();
    
    // Initialize with a predefined set of cards
    explicit Deck(const std::vector<Card>& cards);
    
    // Shuffle the deck using a given random number generator
    void shuffle(std::mt19937& rng);
    
    // Draw a card from the top of the deck
    Card dealCard();
    
    // Reset the deck to a full set of cards
    void reset();
    
    // Get the number of cards remaining in the deck
    size_t cardsRemaining() const;
    
    // Get all cards currently in the deck
    const std::vector<Card>& getCards() const;
    
    // Remove a specific card from the deck (useful for simulations)
    void removeCard(const Card& card);

private:
    std::vector<Card> cards_;
    size_t next_card_index_ = 0;
};

} // namespace poker
