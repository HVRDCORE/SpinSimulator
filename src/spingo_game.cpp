#include "poker/spingo_game.h"
#include "poker/utils.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace poker {

SpinGoGame::SpinGoGame(int num_players, int64_t buy_in, int64_t small_blind, 
                       int64_t big_blind, float prize_multiplier)
    : buy_in_(buy_in), prize_multiplier_(prize_multiplier) {
    
    // Initialize game state
    game_state_ = std::make_unique<GameState>(num_players, buy_in, small_blind, big_blind);
    
    // Calculate prize pool
    prize_pool_ = static_cast<int64_t>(buy_in * num_players * prize_multiplier);
    
    // Seed RNG with current time
    uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
                                        .time_since_epoch().count());
    rng_.seed(seed);
    game_state_->setSeed(seed);
    
    // Initialize blind schedule
    // Example schedule for a Spin & Go
    blind_schedule_ = {
        {10, small_blind, big_blind},      // Level 1: 10 hands
        {10, small_blind * 2, big_blind * 2}, // Level 2: 10 hands
        {10, small_blind * 4, big_blind * 4}, // Level 3: 10 hands
        {10, small_blind * 8, big_blind * 8}, // Level 4: 10 hands
        {10, small_blind * 16, big_blind * 16}, // Level 5: 10 hands
        {10, small_blind * 32, big_blind * 32}, // Level 6: 10 hands
        {10, small_blind * 64, big_blind * 64}, // Level 7: 10 hands
        {10, small_blind * 128, big_blind * 128}, // Level 8: 10 hands
        {10, small_blind * 256, big_blind * 256}, // Level 9: 10 hands
        {-1, small_blind * 512, big_blind * 512}  // Level 10: until end
    };
    
    notify("Spin & Go Tournament Started! Prize Pool: " + formatChips(prize_pool_));
}

void SpinGoGame::play() {
    playToCompletion();
}

void SpinGoGame::playHand() {
    if (isTournamentOver()) {
        notify("Tournament is already over!");
        return;
    }
    
    // Check if we need to update blinds
    updateBlinds();
    
    // Deal hole cards
    game_state_->dealHoleCards();
    notify("New hand started - " + game_state_->toString());
    
    // Play hand until completion
    while (!game_state_->isHandOver()) {
        // Get the current player
        int current_player = game_state_->getCurrentPlayerIndex();
        const auto& player = game_state_->getPlayers()[current_player];
        
        // Get legal actions
        auto legal_actions = game_state_->getLegalActions();
        
        // Choose a default action (fold)
        Action chosen_action = legal_actions[0];
        
        // Apply the action
        game_state_->applyAction(chosen_action);
        
        notify("Player " + player.getName() + " " + chosen_action.toString());
    }
    
    // Hand is over
    auto winners = game_state_->getWinners();
    
    if (winners.size() == 1) {
        const auto& winner = game_state_->getPlayers()[winners[0]];
        notify("Player " + winner.getName() + " wins the hand with " + 
               game_state_->getHandDescription(winners[0]));
    } else if (winners.size() > 1) {
        std::stringstream ss;
        ss << "Pot chopped between: ";
        for (size_t i = 0; i < winners.size(); ++i) {
            if (i > 0) ss << ", ";
            const auto& winner = game_state_->getPlayers()[winners[i]];
            ss << winner.getName() << " (" << game_state_->getHandDescription(winners[i]) << ")";
        }
        notify(ss.str());
    }
    
    // Reset for next hand
    game_state_->resetForNewHand();
    hands_played_++;
}

void SpinGoGame::playToCompletion() {
    while (!isTournamentOver()) {
        playHand();
    }
    
    int winner = getTournamentWinner();
    const auto& winner_player = game_state_->getPlayers()[winner];
    
    notify("Tournament over! Winner: " + winner_player.getName() + 
          " wins " + formatChips(prize_pool_) + "!");
}

const GameState& SpinGoGame::getGameState() const {
    return *game_state_;
}

GameState& SpinGoGame::getGameStateMutable() {
    return *game_state_;
}

bool SpinGoGame::isTournamentOver() const {
    // Tournament is over when only one player has chips
    int players_with_chips = 0;
    for (const auto& player : game_state_->getPlayers()) {
        if (player.getStack() > 0) {
            players_with_chips++;
        }
    }
    
    return players_with_chips <= 1;
}

int SpinGoGame::getTournamentWinner() const {
    if (!isTournamentOver()) {
        throw std::runtime_error("Tournament is not over yet");
    }
    
    // Find the player with chips
    for (size_t i = 0; i < game_state_->getPlayers().size(); ++i) {
        if (game_state_->getPlayers()[i].getStack() > 0) {
            return i;
        }
    }
    
    // Should never reach here
    throw std::runtime_error("No winner found");
}

int64_t SpinGoGame::getPrizePool() const {
    return prize_pool_;
}

void SpinGoGame::setCallback(const GameCallback& callback) {
    callback_ = callback;
}

void SpinGoGame::setSeed(uint64_t seed) {
    rng_.seed(seed);
    game_state_->setSeed(seed);
}

std::string SpinGoGame::toString() const {
    std::stringstream ss;
    
    ss << "Spin & Go Tournament\n";
    ss << "Prize Pool: " << formatChips(prize_pool_) << "\n";
    ss << "Hands Played: " << hands_played_ << "\n";
    ss << "Blind Level: " << (current_blind_level_ + 1) << " (" << 
       formatChips(game_state_->getSmallBlind()) << "/" << 
       formatChips(game_state_->getBigBlind()) << ")\n\n";
    
    ss << game_state_->toString();
    
    return ss.str();
}

void SpinGoGame::updateBlinds() {
    // Check if we need to move to the next blind level
    if (current_blind_level_ < static_cast<int>(blind_schedule_.size()) - 1) {
        const auto& current_level = blind_schedule_[current_blind_level_];
        
        if (current_level.hands > 0 && hands_played_ >= current_level.hands) {
            // Move to next level
            current_blind_level_++;
            const auto& next_level = blind_schedule_[current_blind_level_];
            
            // Get a reference to the mutable game state
            GameState& state = *game_state_;
            
            // Create a new game state with updated blinds
            auto players = state.getPlayers();
            int dealer_pos = state.getDealerPosition();
            
            game_state_ = std::make_unique<GameState>(
                players.size(),
                0, // Initial stack doesn't matter, we'll set it below
                next_level.small_blind,
                next_level.big_blind
            );
            
            // Restore player stacks
            for (size_t i = 0; i < players.size(); ++i) {
                game_state_->getPlayersMutable()[i].adjustStack(players[i].getStack());
            }
            
            // Restore dealer position
            while (game_state_->getDealerPosition() != dealer_pos) {
                game_state_->resetForNewHand();
            }
            
            notify("Blinds increasing to " + formatChips(next_level.small_blind) + 
                  "/" + formatChips(next_level.big_blind));
        }
    }
}

void SpinGoGame::notify(const std::string& message) const {
    if (callback_) {
        callback_(*game_state_, message);
    }
}

} // namespace poker
