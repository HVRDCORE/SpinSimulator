#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace poker {

// Card constants
constexpr int NUM_SUITS = 4;
constexpr int NUM_RANKS = 13;
constexpr int NUM_CARDS = 52;

// Suit definitions
enum class Suit : uint8_t {
    CLUBS = 0,
    DIAMONDS = 1,
    HEARTS = 2,
    SPADES = 3
};

// Rank definitions
enum class Rank : uint8_t {
    TWO = 0,
    THREE = 1,
    FOUR = 2,
    FIVE = 3,
    SIX = 4,
    SEVEN = 5,
    EIGHT = 6,
    NINE = 7,
    TEN = 8,
    JACK = 9,
    QUEEN = 10,
    KING = 11,
    ACE = 12
};

// Hand types
enum class HandType : uint8_t {
    HIGH_CARD = 0,
    PAIR = 1,
    TWO_PAIR = 2,
    THREE_OF_A_KIND = 3,
    STRAIGHT = 4,
    FLUSH = 5,
    FULL_HOUSE = 6,
    FOUR_OF_A_KIND = 7,
    STRAIGHT_FLUSH = 8,
    ROYAL_FLUSH = 9
};

// Game constants for Spin & Go
constexpr int MAX_PLAYERS = 3;      // Spin & Go is typically 3-player
constexpr int NUM_HOLE_CARDS = 2;   // Texas Hold'em has 2 hole cards
constexpr int NUM_COMMUNITY_CARDS = 5; // 5 community cards (flop, turn, river)
constexpr int NUM_BOARD_CARDS = 5;  // Board consists of 5 cards
constexpr int MAX_BETTING_ROUNDS = 4; // Pre-flop, flop, turn, river

// Game stages
enum class GameStage : uint8_t {
    PREFLOP = 0,
    FLOP = 1,
    TURN = 2,
    RIVER = 3,
    SHOWDOWN = 4
};

// Action types
enum class ActionType : uint8_t {
    FOLD = 0,
    CHECK = 1,
    CALL = 2,
    BET = 3,
    RAISE = 4,
    ALL_IN = 5
};

// String representations
extern const std::array<std::string, NUM_SUITS> SUIT_STRINGS;
extern const std::array<std::string, NUM_RANKS> RANK_STRINGS;
extern const std::array<std::string, 10> HAND_TYPE_STRINGS;
extern const std::array<std::string, 6> ACTION_TYPE_STRINGS;
extern const std::array<std::string, 5> GAME_STAGE_STRINGS;

} // namespace poker
