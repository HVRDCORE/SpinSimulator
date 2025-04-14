#include "poker/card.h"
#include <cctype>
#include <stdexcept>

namespace poker {

Card::Card() : id_(-1) {}

Card::Card(Rank rank, Suit suit) 
    : id_(static_cast<int>(rank) + static_cast<int>(suit) * NUM_RANKS) {}

Card::Card(int id) : id_(id) {
    if (id < 0 || id >= NUM_CARDS) {
        throw std::out_of_range("Card ID must be between 0 and 51");
    }
}

Card::Card(const std::string& card_str) {
    if (card_str.length() != 2) {
        throw std::invalid_argument("Card string must be exactly 2 characters");
    }
    
    // Parse rank
    char rank_char = std::toupper(card_str[0]);
    Rank rank;
    
    if (rank_char >= '2' && rank_char <= '9') {
        rank = static_cast<Rank>(rank_char - '2');
    } else if (rank_char == 'T') {
        rank = Rank::TEN;
    } else if (rank_char == 'J') {
        rank = Rank::JACK;
    } else if (rank_char == 'Q') {
        rank = Rank::QUEEN;
    } else if (rank_char == 'K') {
        rank = Rank::KING;
    } else if (rank_char == 'A') {
        rank = Rank::ACE;
    } else {
        throw std::invalid_argument("Invalid rank character");
    }
    
    // Parse suit
    char suit_char = std::tolower(card_str[1]);
    Suit suit;
    
    if (suit_char == 'c') {
        suit = Suit::CLUBS;
    } else if (suit_char == 'd') {
        suit = Suit::DIAMONDS;
    } else if (suit_char == 'h') {
        suit = Suit::HEARTS;
    } else if (suit_char == 's') {
        suit = Suit::SPADES;
    } else {
        throw std::invalid_argument("Invalid suit character");
    }
    
    id_ = static_cast<int>(rank) + static_cast<int>(suit) * NUM_RANKS;
}

Rank Card::getRank() const {
    return static_cast<Rank>(id_ % NUM_RANKS);
}

Suit Card::getSuit() const {
    return static_cast<Suit>(id_ / NUM_RANKS);
}

int Card::getId() const {
    return id_;
}

std::string Card::toString() const {
    if (!isValid()) {
        return "??";
    }
    return RANK_STRINGS[static_cast<int>(getRank())] + 
           SUIT_STRINGS[static_cast<int>(getSuit())];
}

bool Card::operator==(const Card& other) const {
    return id_ == other.id_;
}

bool Card::operator!=(const Card& other) const {
    return id_ != other.id_;
}

bool Card::isValid() const {
    return id_ >= 0 && id_ < NUM_CARDS;
}

std::ostream& operator<<(std::ostream& os, const Card& card) {
    os << card.toString();
    return os;
}

} // namespace poker
