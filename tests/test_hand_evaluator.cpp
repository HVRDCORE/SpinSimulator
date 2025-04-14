#include <gtest/gtest.h>
#include "poker/hand_evaluator.h"

using namespace poker;

class HandEvaluatorTest : public ::testing::Test {
protected:
    HandEvaluator evaluator;
};

TEST_F(HandEvaluatorTest, HighCardHand) {
    // Create a high card hand: 2c, 4h, 7s, 9d, Qc
    std::vector<Card> hand = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::FOUR, Suit::HEARTS),    // 4h
        Card(Rank::SEVEN, Suit::SPADES),   // 7s
        Card(Rank::NINE, Suit::DIAMONDS),  // 9d
        Card(Rank::QUEEN, Suit::CLUBS)     // Qc
    };
    
    uint32_t value = evaluator.evaluate(hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::HIGH_CARD);
    EXPECT_EQ(evaluator.getHandDescription(value), "High Card");
}

TEST_F(HandEvaluatorTest, PairHand) {
    // Create a pair hand: 2c, 2h, 7s, 9d, Qc
    std::vector<Card> hand = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::TWO, Suit::HEARTS),     // 2h
        Card(Rank::SEVEN, Suit::SPADES),   // 7s
        Card(Rank::NINE, Suit::DIAMONDS),  // 9d
        Card(Rank::QUEEN, Suit::CLUBS)     // Qc
    };
    
    uint32_t value = evaluator.evaluate(hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::PAIR);
    EXPECT_EQ(evaluator.getHandDescription(value), "Pair");
}

TEST_F(HandEvaluatorTest, TwoPairHand) {
    // Create a two pair hand: 2c, 2h, 7s, 7d, Qc
    std::vector<Card> hand = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::TWO, Suit::HEARTS),     // 2h
        Card(Rank::SEVEN, Suit::SPADES),   // 7s
        Card(Rank::SEVEN, Suit::DIAMONDS), // 7d
        Card(Rank::QUEEN, Suit::CLUBS)     // Qc
    };
    
    uint32_t value = evaluator.evaluate(hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::TWO_PAIR);
    EXPECT_EQ(evaluator.getHandDescription(value), "Two Pair");
}

TEST_F(HandEvaluatorTest, ThreeOfAKindHand) {
    // Create a three of a kind hand: 2c, 2h, 2s, 9d, Qc
    std::vector<Card> hand = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::TWO, Suit::HEARTS),     // 2h
        Card(Rank::TWO, Suit::SPADES),     // 2s
        Card(Rank::NINE, Suit::DIAMONDS),  // 9d
        Card(Rank::QUEEN, Suit::CLUBS)     // Qc
    };
    
    uint32_t value = evaluator.evaluate(hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::THREE_OF_A_KIND);
    EXPECT_EQ(evaluator.getHandDescription(value), "Three of a Kind");
}

TEST_F(HandEvaluatorTest, StraightHand) {
    // Create a straight hand: 2c, 3h, 4s, 5d, 6c
    std::vector<Card> hand = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::THREE, Suit::HEARTS),   // 3h
        Card(Rank::FOUR, Suit::SPADES),    // 4s
        Card(Rank::FIVE, Suit::DIAMONDS),  // 5d
        Card(Rank::SIX, Suit::CLUBS)       // 6c
    };
    
    uint32_t value = evaluator.evaluate(hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::STRAIGHT);
    EXPECT_EQ(evaluator.getHandDescription(value), "Straight");
}

TEST_F(HandEvaluatorTest, FlushHand) {
    // Create a flush hand: 2c, 4c, 7c, 9c, Qc
    std::vector<Card> hand = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::FOUR, Suit::CLUBS),     // 4c
        Card(Rank::SEVEN, Suit::CLUBS),    // 7c
        Card(Rank::NINE, Suit::CLUBS),     // 9c
        Card(Rank::QUEEN, Suit::CLUBS)     // Qc
    };
    
    uint32_t value = evaluator.evaluate(hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::FLUSH);
    EXPECT_EQ(evaluator.getHandDescription(value), "Flush");
}

TEST_F(HandEvaluatorTest, FullHouseHand) {
    // Create a full house hand: 2c, 2h, 2s, Qd, Qc
    std::vector<Card> hand = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::TWO, Suit::HEARTS),     // 2h
        Card(Rank::TWO, Suit::SPADES),     // 2s
        Card(Rank::QUEEN, Suit::DIAMONDS), // Qd
        Card(Rank::QUEEN, Suit::CLUBS)     // Qc
    };
    
    uint32_t value = evaluator.evaluate(hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::FULL_HOUSE);
    EXPECT_EQ(evaluator.getHandDescription(value), "Full House");
}

TEST_F(HandEvaluatorTest, FourOfAKindHand) {
    // Create a four of a kind hand: 2c, 2h, 2s, 2d, Qc
    std::vector<Card> hand = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::TWO, Suit::HEARTS),     // 2h
        Card(Rank::TWO, Suit::SPADES),     // 2s
        Card(Rank::TWO, Suit::DIAMONDS),   // 2d
        Card(Rank::QUEEN, Suit::CLUBS)     // Qc
    };
    
    uint32_t value = evaluator.evaluate(hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::FOUR_OF_A_KIND);
    EXPECT_EQ(evaluator.getHandDescription(value), "Four of a Kind");
}

TEST_F(HandEvaluatorTest, StraightFlushHand) {
    // Create a straight flush hand: 2c, 3c, 4c, 5c, 6c
    std::vector<Card> hand = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::THREE, Suit::CLUBS),    // 3c
        Card(Rank::FOUR, Suit::CLUBS),     // 4c
        Card(Rank::FIVE, Suit::CLUBS),     // 5c
        Card(Rank::SIX, Suit::CLUBS)       // 6c
    };
    
    uint32_t value = evaluator.evaluate(hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::STRAIGHT_FLUSH);
    EXPECT_EQ(evaluator.getHandDescription(value), "Straight Flush");
}

TEST_F(HandEvaluatorTest, RoyalFlushHand) {
    // Create a royal flush hand: 10c, Jc, Qc, Kc, Ac
    std::vector<Card> hand = {
        Card(Rank::TEN, Suit::CLUBS),      // 10c
        Card(Rank::JACK, Suit::CLUBS),     // Jc
        Card(Rank::QUEEN, Suit::CLUBS),    // Qc
        Card(Rank::KING, Suit::CLUBS),     // Kc
        Card(Rank::ACE, Suit::CLUBS)       // Ac
    };
    
    uint32_t value = evaluator.evaluate(hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::ROYAL_FLUSH);
    EXPECT_EQ(evaluator.getHandDescription(value), "Royal Flush");
}

TEST_F(HandEvaluatorTest, HandComparison) {
    // Compare different hand types
    
    // High card: Ac, 4h, 7s, 9d, Qc
    std::vector<Card> high_card = {
        Card(Rank::ACE, Suit::CLUBS),      // Ac
        Card(Rank::FOUR, Suit::HEARTS),    // 4h
        Card(Rank::SEVEN, Suit::SPADES),   // 7s
        Card(Rank::NINE, Suit::DIAMONDS),  // 9d
        Card(Rank::QUEEN, Suit::CLUBS)     // Qc
    };
    
    // Pair: 2c, 2h, 7s, 9d, Qc
    std::vector<Card> pair = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::TWO, Suit::HEARTS),     // 2h
        Card(Rank::SEVEN, Suit::SPADES),   // 7s
        Card(Rank::NINE, Suit::DIAMONDS),  // 9d
        Card(Rank::QUEEN, Suit::CLUBS)     // Qc
    };
    
    // Royal flush: 10c, Jc, Qc, Kc, Ac
    std::vector<Card> royal_flush = {
        Card(Rank::TEN, Suit::CLUBS),      // 10c
        Card(Rank::JACK, Suit::CLUBS),     // Jc
        Card(Rank::QUEEN, Suit::CLUBS),    // Qc
        Card(Rank::KING, Suit::CLUBS),     // Kc
        Card(Rank::ACE, Suit::CLUBS)       // Ac
    };
    
    uint32_t high_card_value = evaluator.evaluate(high_card);
    uint32_t pair_value = evaluator.evaluate(pair);
    uint32_t royal_flush_value = evaluator.evaluate(royal_flush);
    
    // Royal flush > Pair > High card
    EXPECT_GT(royal_flush_value, pair_value);
    EXPECT_GT(pair_value, high_card_value);
}

TEST_F(HandEvaluatorTest, FindBestHand) {
    // Create 7 cards with a full house as the best 5-card hand
    std::vector<Card> cards = {
        Card(Rank::TWO, Suit::CLUBS),      // 2c
        Card(Rank::TWO, Suit::HEARTS),     // 2h
        Card(Rank::TWO, Suit::SPADES),     // 2s
        Card(Rank::QUEEN, Suit::DIAMONDS), // Qd
        Card(Rank::QUEEN, Suit::CLUBS),    // Qc
        Card(Rank::KING, Suit::HEARTS),    // Kh
        Card(Rank::ACE, Suit::SPADES)      // As
    };
    
    std::vector<Card> best_hand = evaluator.findBestHand(cards);
    
    // Should return 5 cards
    EXPECT_EQ(best_hand.size(), 5);
    
    // Evaluate the returned hand
    uint32_t value = evaluator.evaluate(best_hand);
    EXPECT_EQ(evaluator.getHandType(value), HandType::FULL_HOUSE);
}

TEST_F(HandEvaluatorTest, EvaluateWithHoleAndCommunity) {
    // Create hole cards: Ac, Kc
    std::vector<Card> hole_cards = {
        Card(Rank::ACE, Suit::CLUBS),      // Ac
        Card(Rank::KING, Suit::CLUBS)      // Kc
    };
    
    // Create community cards: Qc, Jc, 10c, 2h, 3d
    std::vector<Card> community_cards = {
        Card(Rank::QUEEN, Suit::CLUBS),    // Qc
        Card(Rank::JACK, Suit::CLUBS),     // Jc
        Card(Rank::TEN, Suit::CLUBS),      // 10c
        Card(Rank::TWO, Suit::HEARTS),     // 2h
        Card(Rank::THREE, Suit::DIAMONDS)  // 3d
    };
    
    uint32_t value = evaluator.evaluate(hole_cards, community_cards);
    EXPECT_EQ(evaluator.getHandType(value), HandType::ROYAL_FLUSH);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
