#include "poker/hand_evaluator.h"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <unordered_map>

namespace poker {

// Hand rank values
constexpr uint32_t ROYAL_FLUSH_BASE = 9000000;
constexpr uint32_t STRAIGHT_FLUSH_BASE = 8000000;
constexpr uint32_t FOUR_OF_A_KIND_BASE = 7000000;
constexpr uint32_t FULL_HOUSE_BASE = 6000000;
constexpr uint32_t FLUSH_BASE = 5000000;
constexpr uint32_t STRAIGHT_BASE = 4000000;
constexpr uint32_t THREE_OF_A_KIND_BASE = 3000000;
constexpr uint32_t TWO_PAIR_BASE = 2000000;
constexpr uint32_t ONE_PAIR_BASE = 1000000;
constexpr uint32_t HIGH_CARD_BASE = 0;

HandEvaluator::HandEvaluator() {
    initLookupTables();
}

void HandEvaluator::initLookupTables() {
    // This is a simplified implementation of lookup tables
    // A full implementation would use perfect hash tables and precomputed values
    // for optimal performance, but that's beyond the scope of this example
    
    // Initialize tables with default values
    flush_lookup_.fill(0);
    unique_lookup_.fill(0);
    
    // Initialize straight patterns
    for (int i = 0; i < 9; ++i) {
        // Straight pattern: 5 consecutive bits
        uint16_t pattern = 0x1F << i;
        unique_lookup_[pattern] = STRAIGHT_BASE + (i + 4); // Value based on high card
    }
    
    // Special case for A-5 straight (Ace is low)
    unique_lookup_[0x100F] = STRAIGHT_BASE + 3; // 5-high straight
}

uint32_t HandEvaluator::evaluate(const std::vector<Card>& cards) const {
    // Check if we have a flush
    bool has_flush = false;
    int suits[4] = {0};
    
    for (const auto& card : cards) {
        suits[static_cast<int>(card.getSuit())]++;
    }
    
    int flush_suit = -1;
    for (int i = 0; i < 4; ++i) {
        if (suits[i] >= 5) {
            has_flush = true;
            flush_suit = i;
            break;
        }
    }
    
    if (has_flush) {
        // Extract cards in the flush suit
        std::vector<Card> flush_cards;
        for (const auto& card : cards) {
            if (static_cast<int>(card.getSuit()) == flush_suit) {
                flush_cards.push_back(card);
            }
        }
        
        return evaluateFlush(flush_cards);
    } else {
        return evaluateNonFlush(cards);
    }
}

uint32_t HandEvaluator::evaluate(const std::vector<Card>& hole_cards,
                                 const std::vector<Card>& community_cards) const {
    std::vector<Card> all_cards;
    all_cards.reserve(hole_cards.size() + community_cards.size());
    all_cards.insert(all_cards.end(), hole_cards.begin(), hole_cards.end());
    all_cards.insert(all_cards.end(), community_cards.begin(), community_cards.end());
    
    return evaluate(all_cards);
}

uint32_t HandEvaluator::evaluateFlush(const std::vector<Card>& cards) const {
    // Create a bitset for ranks in the flush
    std::bitset<13> ranks;
    for (const auto& card : cards) {
        ranks.set(static_cast<int>(card.getRank()));
    }
    
    // Check for straight flush
    uint16_t rank_pattern = static_cast<uint16_t>(ranks.to_ulong() & 0x1FFF);
    
    // Check for straights within the flush
    bool is_straight_flush = false;
    int high_card = -1;
    
    // Check for A-T-J-Q-K straight flush (royal flush)
    if ((rank_pattern & 0x1F00) == 0x1F00) {
        return ROYAL_FLUSH_BASE;
    }
    
    // Check for other straight flushes
    for (int i = 0; i < 9; ++i) {
        uint16_t straight_mask = 0x1F << i;
        if ((rank_pattern & straight_mask) == straight_mask) {
            is_straight_flush = true;
            high_card = i + 4; // High card of the straight
            break;
        }
    }
    
    // Special case: A-2-3-4-5 straight flush
    if ((rank_pattern & 0x100F) == 0x100F) {
        is_straight_flush = true;
        high_card = 3; // 5-high straight
    }
    
    if (is_straight_flush) {
        return STRAIGHT_FLUSH_BASE + high_card;
    }
    
    // Regular flush - use the 5 highest cards
    std::vector<int> flush_ranks;
    for (int i = NUM_RANKS - 1; i >= 0; --i) {
        if (ranks[i]) {
            flush_ranks.push_back(i);
            if (flush_ranks.size() == 5) break;
        }
    }
    
    // Calculate hand value
    uint32_t value = FLUSH_BASE;
    for (int i = 0; i < 5; ++i) {
        value += flush_ranks[i] * static_cast<uint32_t>(pow(13, 4 - i));
    }
    
    return value;
}

uint32_t HandEvaluator::evaluateNonFlush(const std::vector<Card>& cards) const {
    // Count occurrences of each rank
    std::array<int, NUM_RANKS> rank_counts = {0};
    for (const auto& card : cards) {
        rank_counts[static_cast<int>(card.getRank())]++;
    }
    
    // Extract rank groups (quads, trips, pairs)
    std::vector<int> quads, trips, pairs, singles;
    for (int i = NUM_RANKS - 1; i >= 0; --i) {
        if (rank_counts[i] == 4) quads.push_back(i);
        else if (rank_counts[i] == 3) trips.push_back(i);
        else if (rank_counts[i] == 2) pairs.push_back(i);
        else if (rank_counts[i] == 1) singles.push_back(i);
    }
    
    // Four of a kind
    if (!quads.empty()) {
        int kicker = singles.empty() ? 
                    (!trips.empty() ? trips[0] : 
                     (!pairs.empty() ? pairs[0] : -1)) : 
                    singles[0];
        return FOUR_OF_A_KIND_BASE + quads[0] * 13 + kicker;
    }
    
    // Full house
    if (!trips.empty() && (!pairs.empty() || trips.size() > 1)) {
        int trip_rank = trips[0];
        int pair_rank = trips.size() > 1 ? trips[1] : pairs[0];
        return FULL_HOUSE_BASE + trip_rank * 13 + pair_rank;
    }
    
    // Check for straights
    std::bitset<13> rank_bits;
    for (int i = 0; i < NUM_RANKS; ++i) {
        if (rank_counts[i] > 0) {
            rank_bits.set(i);
        }
    }
    
    uint16_t rank_pattern = static_cast<uint16_t>(rank_bits.to_ulong() & 0x1FFF);
    
    // Check for straights
    for (int i = 8; i >= 0; --i) {
        uint16_t straight_mask = 0x1F << i;
        if ((rank_pattern & straight_mask) == straight_mask) {
            return STRAIGHT_BASE + (i + 4);
        }
    }
    
    // A-2-3-4-5 straight
    if ((rank_pattern & 0x100F) == 0x100F) {
        return STRAIGHT_BASE + 3;  // 5-high straight
    }
    
    // Three of a kind
    if (!trips.empty()) {
        int kickers[2] = {-1, -1};
        for (int i = 0; i < std::min(2, static_cast<int>(singles.size())); ++i) {
            kickers[i] = singles[i];
        }
        return THREE_OF_A_KIND_BASE + trips[0] * 169 + 
               kickers[0] * 13 + kickers[1];
    }
    
    // Two pair
    if (pairs.size() >= 2) {
        int kicker = singles.empty() ? 
                   (pairs.size() > 2 ? pairs[2] : -1) : 
                   singles[0];
        return TWO_PAIR_BASE + pairs[0] * 169 + pairs[1] * 13 + kicker;
    }
    
    // One pair
    if (pairs.size() == 1) {
        int kickers[3] = {-1, -1, -1};
        for (int i = 0; i < std::min(3, static_cast<int>(singles.size())); ++i) {
            kickers[i] = singles[i];
        }
        return ONE_PAIR_BASE + pairs[0] * 2197 + 
               kickers[0] * 169 + kickers[1] * 13 + kickers[2];
    }
    
    // High card
    int kickers[5] = {-1, -1, -1, -1, -1};
    for (int i = 0; i < std::min(5, static_cast<int>(singles.size())); ++i) {
        kickers[i] = singles[i];
    }
    
    return HIGH_CARD_BASE + 
           kickers[0] * 371293 + kickers[1] * 28561 + 
           kickers[2] * 2197 + kickers[3] * 169 + kickers[4];
}

HandType HandEvaluator::getHandType(uint32_t hand_value) const {
    if (hand_value >= ROYAL_FLUSH_BASE) return HandType::ROYAL_FLUSH;
    if (hand_value >= STRAIGHT_FLUSH_BASE) return HandType::STRAIGHT_FLUSH;
    if (hand_value >= FOUR_OF_A_KIND_BASE) return HandType::FOUR_OF_A_KIND;
    if (hand_value >= FULL_HOUSE_BASE) return HandType::FULL_HOUSE;
    if (hand_value >= FLUSH_BASE) return HandType::FLUSH;
    if (hand_value >= STRAIGHT_BASE) return HandType::STRAIGHT;
    if (hand_value >= THREE_OF_A_KIND_BASE) return HandType::THREE_OF_A_KIND;
    if (hand_value >= TWO_PAIR_BASE) return HandType::TWO_PAIR;
    if (hand_value >= ONE_PAIR_BASE) return HandType::PAIR;
    return HandType::HIGH_CARD;
}

std::string HandEvaluator::getHandDescription(uint32_t hand_value) const {
    HandType type = getHandType(hand_value);
    return HAND_TYPE_STRINGS[static_cast<int>(type)];
}

std::vector<Card> HandEvaluator::findBestHand(const std::vector<Card>& cards) const {
    // This is a computationally expensive operation, so for simplicity
    // we'll just check all 5-card combinations
    
    if (cards.size() <= 5) {
        return cards;
    }
    
    std::vector<std::vector<Card>> combinations;
    std::vector<Card> current_combo(5);
    
    // Generate all 5-card combinations using a recursive helper function
    auto generateCombinations = [&](auto&& self, size_t start, size_t depth) -> void {
        if (depth == 5) {
            combinations.push_back(current_combo);
            return;
        }
        
        for (size_t i = start; i <= cards.size() - (5 - depth); ++i) {
            current_combo[depth] = cards[i];
            self(self, i + 1, depth + 1);
        }
    };
    
    generateCombinations(generateCombinations, 0, 0);
    
    // Find the best hand
    uint32_t best_value = 0;
    size_t best_index = 0;
    
    for (size_t i = 0; i < combinations.size(); ++i) {
        uint32_t value = evaluate(combinations[i]);
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    
    return combinations[best_index];
}

} // namespace poker
