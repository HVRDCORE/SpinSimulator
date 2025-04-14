#include "poker/utils.h"
#include "poker/constants.h"

namespace poker {

// Initialize string representations
const std::array<std::string, NUM_SUITS> SUIT_STRINGS = {
    "c", "d", "h", "s"
};

const std::array<std::string, NUM_RANKS> RANK_STRINGS = {
    "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"
};

const std::array<std::string, 10> HAND_TYPE_STRINGS = {
    "High Card",
    "Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
    "Royal Flush"
};

const std::array<std::string, 6> ACTION_TYPE_STRINGS = {
    "Fold",
    "Check",
    "Call",
    "Bet",
    "Raise",
    "All-In"
};

const std::array<std::string, 5> GAME_STAGE_STRINGS = {
    "Pre-Flop",
    "Flop",
    "Turn",
    "River",
    "Showdown"
};

// Utility functions implementation
std::string formatChips(int64_t chips) {
    std::stringstream ss;
    if (chips >= 1000000) {
        ss << std::fixed << std::setprecision(1) << (chips / 1000000.0) << "M";
    } else if (chips >= 1000) {
        ss << std::fixed << std::setprecision(1) << (chips / 1000.0) << "K";
    } else {
        ss << chips;
    }
    return ss.str();
}

} // namespace poker
