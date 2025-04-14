#include <gtest/gtest.h>
#include "poker/game_state.h"
#include "poker/spingo_game.h"

using namespace poker;

class GameStateTest : public ::testing::Test {
protected:
    GameState game_state;
    
    GameStateTest() : game_state(3, 1000, 10, 20) {}
    
    void SetUp() override {
        // Use a fixed seed for deterministic tests
        game_state.setSeed(12345);
    }
};

TEST_F(GameStateTest, Initialization) {
    EXPECT_EQ(game_state.getCurrentStage(), GameStage::PREFLOP);
    EXPECT_EQ(game_state.getPot(), 0);
    EXPECT_EQ(game_state.getCommunityCards().size(), 0);
    EXPECT_EQ(game_state.getPlayers().size(), 3);
    EXPECT_EQ(game_state.getSmallBlind(), 10);
    EXPECT_EQ(game_state.getBigBlind(), 20);
    
    // Check initial player stacks
    for (const auto& player : game_state.getPlayers()) {
        EXPECT_EQ(player.getStack(), 1000);
    }
}

TEST_F(GameStateTest, DealCards) {
    // Deal hole cards
    game_state.dealHoleCards();
    
    // Check that each player has 2 cards
    for (const auto& player : game_state.getPlayers()) {
        EXPECT_EQ(player.getHoleCards().size(), NUM_HOLE_CARDS);
        
        // Each card should be valid
        for (const auto& card : player.getHoleCards()) {
            EXPECT_TRUE(card.isValid());
        }
    }
    
    // Deal flop
    game_state.dealFlop();
    EXPECT_EQ(game_state.getCurrentStage(), GameStage::FLOP);
    EXPECT_EQ(game_state.getCommunityCards().size(), 3);
    
    // Deal turn
    game_state.dealTurn();
    EXPECT_EQ(game_state.getCurrentStage(), GameStage::TURN);
    EXPECT_EQ(game_state.getCommunityCards().size(), 4);
    
    // Deal river
    game_state.dealRiver();
    EXPECT_EQ(game_state.getCurrentStage(), GameStage::RIVER);
    EXPECT_EQ(game_state.getCommunityCards().size(), 5);
}

TEST_F(GameStateTest, PlayerActions) {
    // Deal hole cards
    game_state.dealHoleCards();
    
    // Check initial state
    int initial_player = game_state.getCurrentPlayerIndex();
    EXPECT_FALSE(game_state.isHandOver());
    
    // Get legal actions
    auto legal_actions = game_state.getLegalActions();
    EXPECT_GT(legal_actions.size(), 0);
    
    // Apply an action
    Action action = legal_actions[0]; // Use first legal action
    game_state.applyAction(action);
    
    // Check that the current player has changed
    EXPECT_NE(game_state.getCurrentPlayerIndex(), initial_player);
}

TEST_F(GameStateTest, HandEvaluation) {
    // Deal hole cards
    game_state.dealHoleCards();
    
    // Deal all community cards
    game_state.dealFlop();
    game_state.dealTurn();
    game_state.dealRiver();
    
    // Check hand values
    for (size_t i = 0; i < game_state.getPlayers().size(); ++i) {
        if (!game_state.getPlayers()[i].hasFolded()) {
            uint32_t hand_value = game_state.getHandValue(i);
            EXPECT_GT(hand_value, 0);
            
            std::string hand_desc = game_state.getHandDescription(i);
            EXPECT_FALSE(hand_desc.empty());
        }
    }
}

TEST_F(GameStateTest, ResetForNewHand) {
    // Deal hole cards and advance to flop
    game_state.dealHoleCards();
    game_state.dealFlop();
    
    // Reset for new hand
    game_state.resetForNewHand();
    
    // Check that the state has been reset
    EXPECT_EQ(game_state.getCurrentStage(), GameStage::PREFLOP);
    EXPECT_EQ(game_state.getCommunityCards().size(), 0);
    
    // Players should still have their stacks
    for (const auto& player : game_state.getPlayers()) {
        EXPECT_FALSE(player.hasFolded());
        EXPECT_FALSE(player.isAllIn());
    }
}

class SpinGoGameTest : public ::testing::Test {
protected:
    SpinGoGame game;
    
    SpinGoGameTest() : game(3, 1000, 10, 20, 2.0f) {}
    
    void SetUp() override {
        // Use a fixed seed for deterministic tests
        game.setSeed(12345);
    }
};

TEST_F(SpinGoGameTest, Initialization) {
    EXPECT_FALSE(game.isTournamentOver());
    EXPECT_EQ(game.getPrizePool(), 6000); // 3 players * 1000 buy-in * 2.0 multiplier
    
    const auto& state = game.getGameState();
    EXPECT_EQ(state.getPlayers().size(), 3);
    EXPECT_EQ(state.getSmallBlind(), 10);
    EXPECT_EQ(state.getBigBlind(), 20);
}

TEST_F(SpinGoGameTest, PlayHand) {
    // Play a hand
    game.playHand();
    
    // Check that the game state has changed
    const auto& state = game.getGameState();
    
    // The hand should be over and we should have reset for the next hand
    EXPECT_EQ(state.getCurrentStage(), GameStage::PREFLOP);
    EXPECT_EQ(state.getCommunityCards().size(), 0);
}

TEST_F(SpinGoGameTest, PlayToCompletion) {
    // Set a callback to observe game progress
    int hands_played = 0;
    game.setCallback([&](const GameState& state, const std::string& message) {
        if (message.find("New hand started") != std::string::npos) {
            hands_played++;
        }
    });
    
    // Play the tournament to completion
    game.playToCompletion();
    
    // Tournament should be over
    EXPECT_TRUE(game.isTournamentOver());
    
    // Check that we have a winner
    int winner = game.getTournamentWinner();
    EXPECT_GE(winner, 0);
    EXPECT_LT(winner, 3);
    
    // Winner should have all the chips
    const auto& state = game.getGameState();
    EXPECT_EQ(state.getPlayers()[winner].getStack(), 3000); // 3 players * 1000 buy-in
    
    // Other players should have 0 chips
    for (size_t i = 0; i < state.getPlayers().size(); ++i) {
        if (i != static_cast<size_t>(winner)) {
            EXPECT_EQ(state.getPlayers()[i].getStack(), 0);
        }
    }
    
    // Should have played at least one hand
    EXPECT_GT(hands_played, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
