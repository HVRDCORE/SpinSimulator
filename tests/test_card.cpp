#include <gtest/gtest.h>
#include "poker/card.h"

using namespace poker;

TEST(CardTest, DefaultConstructor) {
    Card card;
    EXPECT_FALSE(card.isValid());
}

TEST(CardTest, RankSuitConstructor) {
    Card card(Rank::ACE, Suit::SPADES);
    EXPECT_EQ(card.getRank(), Rank::ACE);
    EXPECT_EQ(card.getSuit(), Suit::SPADES);
    EXPECT_EQ(card.getId(), 51); // ACE (12) + SPADES (3) * 13
    EXPECT_TRUE(card.isValid());
}

TEST(CardTest, IdConstructor) {
    Card card(0); // 2 of Clubs
    EXPECT_EQ(card.getRank(), Rank::TWO);
    EXPECT_EQ(card.getSuit(), Suit::CLUBS);
    EXPECT_TRUE(card.isValid());
    
    Card card2(51); // Ace of Spades
    EXPECT_EQ(card2.getRank(), Rank::ACE);
    EXPECT_EQ(card2.getSuit(), Suit::SPADES);
    EXPECT_TRUE(card2.isValid());
    
    // Test invalid ID
    EXPECT_THROW(Card(-1), std::out_of_range);
    EXPECT_THROW(Card(52), std::out_of_range);
}

TEST(CardTest, StringConstructor) {
    Card card("As"); // Ace of Spades
    EXPECT_EQ(card.getRank(), Rank::ACE);
    EXPECT_EQ(card.getSuit(), Suit::SPADES);
    EXPECT_EQ(card.toString(), "As");
    EXPECT_TRUE(card.isValid());
    
    Card card2("2c"); // 2 of Clubs
    EXPECT_EQ(card2.getRank(), Rank::TWO);
    EXPECT_EQ(card2.getSuit(), Suit::CLUBS);
    EXPECT_EQ(card2.toString(), "2c");
    EXPECT_TRUE(card2.isValid());
    
    Card card3("Td"); // 10 of Diamonds
    EXPECT_EQ(card3.getRank(), Rank::TEN);
    EXPECT_EQ(card3.getSuit(), Suit::DIAMONDS);
    EXPECT_EQ(card3.toString(), "Td");
    EXPECT_TRUE(card3.isValid());
    
    // Test case-insensitivity
    Card card4("kH"); // King of Hearts
    EXPECT_EQ(card4.getRank(), Rank::KING);
    EXPECT_EQ(card4.getSuit(), Suit::HEARTS);
    EXPECT_EQ(card4.toString(), "Kh");
    EXPECT_TRUE(card4.isValid());
    
    // Test invalid strings
    EXPECT_THROW(Card(""), std::invalid_argument);
    EXPECT_THROW(Card("A"), std::invalid_argument);
    EXPECT_THROW(Card("As1"), std::invalid_argument);
    EXPECT_THROW(Card("Ax"), std::invalid_argument);
    EXPECT_THROW(Card("1s"), std::invalid_argument);
}

TEST(CardTest, ComparisonOperators) {
    Card ace_spades(Rank::ACE, Suit::SPADES);
    Card ace_spades2(Rank::ACE, Suit::SPADES);
    Card king_spades(Rank::KING, Suit::SPADES);
    Card ace_hearts(Rank::ACE, Suit::HEARTS);
    
    EXPECT_TRUE(ace_spades == ace_spades2);
    EXPECT_FALSE(ace_spades == king_spades);
    EXPECT_FALSE(ace_spades == ace_hearts);
    
    EXPECT_FALSE(ace_spades != ace_spades2);
    EXPECT_TRUE(ace_spades != king_spades);
    EXPECT_TRUE(ace_spades != ace_hearts);
}

TEST(CardTest, ToString) {
    Card two_clubs(Rank::TWO, Suit::CLUBS);
    EXPECT_EQ(two_clubs.toString(), "2c");
    
    Card jack_diamonds(Rank::JACK, Suit::DIAMONDS);
    EXPECT_EQ(jack_diamonds.toString(), "Jd");
    
    Card queen_hearts(Rank::QUEEN, Suit::HEARTS);
    EXPECT_EQ(queen_hearts.toString(), "Qh");
    
    Card king_spades(Rank::KING, Suit::SPADES);
    EXPECT_EQ(king_spades.toString(), "Ks");
    
    Card ace_spades(Rank::ACE, Suit::SPADES);
    EXPECT_EQ(ace_spades.toString(), "As");
    
    // Test invalid card
    Card invalid;
    EXPECT_EQ(invalid.toString(), "??");
}

TEST(CardTest, StreamOperator) {
    Card ace_spades(Rank::ACE, Suit::SPADES);
    std::stringstream ss;
    ss << ace_spades;
    EXPECT_EQ(ss.str(), "As");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
