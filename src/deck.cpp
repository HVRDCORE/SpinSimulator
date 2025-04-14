#include "poker/deck.h"
#include <algorithm>
#include <stdexcept>

namespace poker {

Deck::Deck() {
    reset();
}

Deck::Deck(const std::vector<Card>& cards) : cards_(cards), next_card_index_(0) {}

void Deck::shuffle(std::mt19937& rng) {
    std::shuffle(cards_.begin(), cards_.end(), rng);
    next_card_index_ = 0;
}

Card Deck::dealCard() {
    if (next_card_index_ >= cards_.size()) {
        throw std::runtime_error("No cards left in the deck");
    }
    return cards_[next_card_index_++];
}

void Deck::reset() {
    cards_.clear();
    cards_.reserve(NUM_CARDS);
    
    for (int suit = 0; suit < NUM_SUITS; ++suit) {
        for (int rank = 0; rank < NUM_RANKS; ++rank) {
            cards_.emplace_back(static_cast<Rank>(rank), static_cast<Suit>(suit));
        }
    }
    
    next_card_index_ = 0;
}

size_t Deck::cardsRemaining() const {
    return cards_.size() - next_card_index_;
}

const std::vector<Card>& Deck::getCards() const {
    return cards_;
}

void Deck::removeCard(const Card& card) {
    // Search for the card in the remaining cards
    for (size_t i = next_card_index_; i < cards_.size(); ++i) {
        if (cards_[i] == card) {
            // Swap the card with the last one and remove it
            std::swap(cards_[i], cards_.back());
            cards_.pop_back();
            return;
        }
    }
    
    throw std::runtime_error("Card not found in deck");
}

} // namespace poker
