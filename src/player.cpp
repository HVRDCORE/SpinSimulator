#include "poker/player.h"
#include "poker/utils.h"
#include <sstream>
#include <stdexcept>

namespace poker {

Player::Player(int id, int64_t initial_stack, const std::string& name)
    : id_(id), name_(name.empty() ? "Player " + std::to_string(id) : name),
      stack_(initial_stack), has_cards_(false) {}

int Player::getId() const {
    return id_;
}

const std::string& Player::getName() const {
    return name_;
}

void Player::setHoleCards(const std::array<Card, NUM_HOLE_CARDS>& cards) {
    hole_cards_ = cards;
    has_cards_ = true;
}

const std::array<Card, NUM_HOLE_CARDS>& Player::getHoleCards() const {
    return hole_cards_;
}

int64_t Player::getStack() const {
    return stack_;
}

void Player::adjustStack(int64_t amount) {
    if (amount < 0 && -amount > stack_) {
        throw std::invalid_argument("Cannot remove more chips than in stack");
    }
    stack_ += amount;
}

int64_t Player::getCurrentBet() const {
    return current_bet_;
}

void Player::setCurrentBet(int64_t amount) {
    if (amount < 0) {
        throw std::invalid_argument("Bet amount cannot be negative");
    }
    current_bet_ = amount;
}

void Player::resetCurrentBet() {
    current_bet_ = 0;
}

bool Player::isAllIn() const {
    return has_cards_ && !has_folded_ && stack_ == 0;
}

bool Player::hasFolded() const {
    return has_folded_;
}

void Player::setFolded(bool folded) {
    has_folded_ = folded;
}

bool Player::isActive() const {
    return has_cards_ && !has_folded_;
}

void Player::resetForNewHand() {
    has_folded_ = false;
    has_cards_ = false;
    current_bet_ = 0;
}

std::string Player::toString() const {
    std::stringstream ss;
    ss << name_ << " (" << formatChips(stack_) << ")";
    
    if (has_cards_) {
        ss << " [";
        for (size_t i = 0; i < hole_cards_.size(); ++i) {
            if (i > 0) ss << " ";
            ss << hole_cards_[i];
        }
        ss << "]";
    }
    
    if (has_folded_) {
        ss << " (folded)";
    } else if (isAllIn()) {
        ss << " (all-in)";
    }
    
    return ss.str();
}

} // namespace poker
