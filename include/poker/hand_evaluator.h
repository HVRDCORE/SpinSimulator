#pragma once

#include "poker/card.h"
#include <array>
#include <vector>

namespace poker {

/**
 * Fast poker hand evaluator for Texas Hold'em hands.
 * Uses a lookup-based approach for efficiency.
 */
class HandEvaluator {
public:
    // Initialize the evaluator with precomputed tables
    HandEvaluator();
    
    // Evaluates a 5-7 card poker hand and returns a numeric hand strength
    // Higher value = stronger hand
    uint32_t evaluate(const std::vector<Card>& cards) const;
    
    // Overload for array-based cards (e.g., when working with hole and community cards)
    uint32_t evaluate(const std::vector<Card>& hole_cards,
                     const std::vector<Card>& community_cards) const;
    
    // Get the hand type (pair, straight, etc.) from a hand value
    HandType getHandType(uint32_t hand_value) const;
    
    // Get a textual description of the hand
    std::string getHandDescription(uint32_t hand_value) const;
    
    // Find the best 5-card hand from the given cards
    std::vector<Card> findBestHand(const std::vector<Card>& cards) const;

private:
    // Lookup tables for hand evaluation
    std::array<uint32_t, 8192> flush_lookup_;
    std::array<uint32_t, 7937> unique_lookup_;
    
    // Initialize lookup tables
    void initLookupTables();
    
    // Helper functions for evaluation
    uint32_t evaluateFlush(const std::vector<Card>& cards) const;
    uint32_t evaluateNonFlush(const std::vector<Card>& cards) const;
};

} // namespace poker
