#include "poker/game_state.h"
#include "poker/utils.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace poker {

GameState::GameState(int num_players, int64_t initial_stack, int64_t small_blind, int64_t big_blind)
    : small_blind_(small_blind), big_blind_(big_blind) {
    
    if (num_players < 2 || num_players > MAX_PLAYERS) {
        throw std::invalid_argument("Number of players must be between 2 and " + 
                                    std::to_string(MAX_PLAYERS));
    }
    
    // Initialize players
    players_.reserve(num_players);
    for (int i = 0; i < num_players; ++i) {
        players_.emplace_back(i, initial_stack);
    }
    
    // Seed the RNG with current time
    seed_ = static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
                                 .time_since_epoch().count());
    rng_.seed(seed_);
    
    resetForNewHand();
}

void GameState::resetForNewHand() {
    // Reset game state
    community_cards_.clear();
    pot_ = 0;
    current_min_bet_ = big_blind_;
    min_raise_ = big_blind_;
    current_stage_ = GameStage::PREFLOP;
    action_history_.clear();
    
    // Reset players
    for (auto& player : players_) {
        player.resetForNewHand();
    }
    
    // Move dealer button
    dealer_position_ = (dealer_position_ + 1) % players_.size();
    
    // Initialize deck and shuffle
    deck_ = Deck();
    deck_.shuffle(rng_);
    
    // Post blinds
    postBlinds();
    
    // Set first player to act (after big blind)
    current_player_index_ = (dealer_position_ + 3) % players_.size();
    if (players_.size() == 2) { // Heads-up
        current_player_index_ = (dealer_position_ + 1) % players_.size();
    }
    
    // Make sure the first player is an active player
    while (!players_[current_player_index_].isActive() && !isHandOver()) {
        advanceToNextPlayer();
    }
}

void GameState::dealHoleCards() {
    for (auto& player : players_) {
        if (player.getStack() > 0) {
            std::array<Card, NUM_HOLE_CARDS> cards;
            for (int i = 0; i < NUM_HOLE_CARDS; ++i) {
                cards[i] = deck_.dealCard();
            }
            player.setHoleCards(cards);
        }
    }
}

void GameState::dealFlop() {
    if (current_stage_ != GameStage::PREFLOP) {
        throw std::runtime_error("Cannot deal flop: wrong game stage");
    }
    
    // Burn a card
    deck_.dealCard();
    
    // Deal 3 community cards
    for (int i = 0; i < 3; ++i) {
        community_cards_.push_back(deck_.dealCard());
    }
    
    current_stage_ = GameStage::FLOP;
    current_min_bet_ = big_blind_;
    min_raise_ = big_blind_;
    
    // Move bets to pot
    moveBetsToPot();
    
    // Find first active player after dealer
    current_player_index_ = (dealer_position_ + 1) % players_.size();
    while (!players_[current_player_index_].isActive() && !isHandOver()) {
        advanceToNextPlayer();
    }
}

void GameState::dealTurn() {
    if (current_stage_ != GameStage::FLOP) {
        throw std::runtime_error("Cannot deal turn: wrong game stage");
    }
    
    // Burn a card
    deck_.dealCard();
    
    // Deal turn card
    community_cards_.push_back(deck_.dealCard());
    
    current_stage_ = GameStage::TURN;
    current_min_bet_ = big_blind_;
    min_raise_ = big_blind_;
    
    // Move bets to pot
    moveBetsToPot();
    
    // Find first active player after dealer
    current_player_index_ = (dealer_position_ + 1) % players_.size();
    while (!players_[current_player_index_].isActive() && !isHandOver()) {
        advanceToNextPlayer();
    }
}

void GameState::dealRiver() {
    if (current_stage_ != GameStage::TURN) {
        throw std::runtime_error("Cannot deal river: wrong game stage");
    }
    
    // Burn a card
    deck_.dealCard();
    
    // Deal river card
    community_cards_.push_back(deck_.dealCard());
    
    current_stage_ = GameStage::RIVER;
    current_min_bet_ = big_blind_;
    min_raise_ = big_blind_;
    
    // Move bets to pot
    moveBetsToPot();
    
    // Find first active player after dealer
    current_player_index_ = (dealer_position_ + 1) % players_.size();
    while (!players_[current_player_index_].isActive() && !isHandOver()) {
        advanceToNextPlayer();
    }
}

void GameState::advanceStage() {
    switch (current_stage_) {
        case GameStage::PREFLOP:
            dealFlop();
            break;
        case GameStage::FLOP:
            dealTurn();
            break;
        case GameStage::TURN:
            dealRiver();
            break;
        case GameStage::RIVER:
            current_stage_ = GameStage::SHOWDOWN;
            distributePot();
            break;
        case GameStage::SHOWDOWN:
            resetForNewHand();
            break;
    }
}

GameStage GameState::getCurrentStage() const {
    return current_stage_;
}

const std::vector<Card>& GameState::getCommunityCards() const {
    return community_cards_;
}

const std::vector<Player>& GameState::getPlayers() const {
    return players_;
}

std::vector<Player>& GameState::getPlayersMutable() {
    return players_;
}

int64_t GameState::getPot() const {
    return pot_;
}

int GameState::getCurrentPlayerIndex() const {
    return current_player_index_;
}

int GameState::getDealerPosition() const {
    return dealer_position_;
}

int64_t GameState::getSmallBlind() const {
    return small_blind_;
}

int64_t GameState::getBigBlind() const {
    return big_blind_;
}

int64_t GameState::getCurrentMinBet() const {
    return current_min_bet_;
}

int64_t GameState::getMinRaise() const {
    return min_raise_;
}

void GameState::applyAction(const Action& action) {
    // Make sure it's a legal action
    bool is_legal = false;
    auto legal_actions = getLegalActions();
    for (const auto& legal_action : legal_actions) {
        if (action.getType() == legal_action.getType() &&
            (action.getType() == ActionType::FOLD || action.getType() == ActionType::CHECK ||
             action.getAmount() == legal_action.getAmount())) {
            is_legal = true;
            break;
        }
    }
    
    if (!is_legal) {
        throw std::invalid_argument("Illegal action: " + action.toString());
    }
    
    // Current player
    Player& player = players_[current_player_index_];
    
    // Apply the action
    switch (action.getType()) {
        case ActionType::FOLD:
            player.setFolded(true);
            break;
            
        case ActionType::CHECK:
            // No changes to state
            break;
            
        case ActionType::CALL: {
            int64_t call_amount = action.getAmount();
            
            // Player is going all-in with less than the full call amount
            if (call_amount > player.getStack()) {
                call_amount = player.getStack();
            }
            
            player.adjustStack(-call_amount);
            player.setCurrentBet(player.getCurrentBet() + call_amount);
            break;
        }
            
        case ActionType::BET: {
            int64_t bet_amount = action.getAmount();
            player.adjustStack(-bet_amount);
            player.setCurrentBet(player.getCurrentBet() + bet_amount);
            current_min_bet_ = player.getCurrentBet();
            min_raise_ = bet_amount;
            last_aggressor_ = current_player_index_;
            break;
        }
            
        case ActionType::RAISE: {
            int64_t raise_amount = action.getAmount();
            player.adjustStack(-raise_amount);
            player.setCurrentBet(player.getCurrentBet() + raise_amount);
            min_raise_ = player.getCurrentBet() - current_min_bet_;
            current_min_bet_ = player.getCurrentBet();
            last_aggressor_ = current_player_index_;
            break;
        }
            
        case ActionType::ALL_IN: {
            int64_t all_in_amount = player.getStack();
            player.adjustStack(-all_in_amount);
            player.setCurrentBet(player.getCurrentBet() + all_in_amount);
            
            // If this is a raise, update the minimum raise
            if (player.getCurrentBet() > current_min_bet_) {
                int64_t raise_size = player.getCurrentBet() - current_min_bet_;
                if (raise_size >= min_raise_) {
                    min_raise_ = raise_size;
                }
                current_min_bet_ = player.getCurrentBet();
                last_aggressor_ = current_player_index_;
            }
            break;
        }
    }
    
    // Record the action
    action_history_.push_back(action);
    
    // If the hand is over after this action, distribute the pot
    if (getPlayersInHand() == 1) {
        // Find the only player still in the hand
        int winner = -1;
        for (size_t i = 0; i < players_.size(); ++i) {
            if (players_[i].isActive()) {
                winner = i;
                break;
            }
        }
        
        if (winner != -1) {
            // Move all bets to pot
            moveBetsToPot();
            
            // Award pot to winner
            players_[winner].adjustStack(pot_);
            pot_ = 0;
            
            // Move to showdown to end the hand
            current_stage_ = GameStage::SHOWDOWN;
            return;
        }
    }
    
    // Check if betting round is over
    if (isBettingRoundOver()) {
        if (current_stage_ == GameStage::RIVER) {
            // Move to showdown
            moveBetsToPot();
            current_stage_ = GameStage::SHOWDOWN;
            distributePot();
        } else {
            // Move to next stage
            advanceStage();
        }
        return;
    }
    
    // Move to next player
    advanceToNextPlayer();
}

std::vector<Action> GameState::getLegalActions() const {
    std::vector<Action> legal_actions;
    
    // If the hand is over, no actions are legal
    if (isHandOver()) {
        return legal_actions;
    }
    
    const Player& player = players_[current_player_index_];
    int64_t player_stack = player.getStack();
    int64_t player_bet = player.getCurrentBet();
    
    // Fold is always legal
    legal_actions.push_back(Action::fold());
    
    // Check is legal if no bet has been made or player has matched the current bet
    if (current_min_bet_ == player_bet) {
        legal_actions.push_back(Action::check());
    }
    
    // Call is legal if there's a bet to call and player has enough chips
    if (current_min_bet_ > player_bet) {
        int64_t call_amount = current_min_bet_ - player_bet;
        if (call_amount >= player_stack) {
            // All-in call
            legal_actions.push_back(Action::allIn(player_stack));
        } else {
            legal_actions.push_back(Action::call(call_amount));
        }
    }
    
    // Bet is legal if no bet has been made and player has enough chips
    if (current_min_bet_ == 0 && player_stack > 0) {
        int64_t min_bet = big_blind_;
        if (min_bet >= player_stack) {
            // All-in bet
            legal_actions.push_back(Action::allIn(player_stack));
        } else {
            legal_actions.push_back(Action::bet(min_bet));
            
            // Also add all-in if it's different from min bet
            if (player_stack > min_bet) {
                legal_actions.push_back(Action::allIn(player_stack));
            }
        }
    }
    
    // Raise is legal if there's a bet to raise and player has enough chips
    if (current_min_bet_ > 0 && player_stack > (current_min_bet_ - player_bet)) {
        int64_t min_raise_to = current_min_bet_ + min_raise_;
        int64_t raise_amount = min_raise_to - player_bet;
        
        if (raise_amount >= player_stack) {
            // All-in raise
            legal_actions.push_back(Action::allIn(player_stack));
        } else {
            legal_actions.push_back(Action::raise(raise_amount));
            
            // Also add all-in if it's different from min raise
            if (player_stack > raise_amount) {
                legal_actions.push_back(Action::allIn(player_stack));
            }
        }
    }
    
    return legal_actions;
}

bool GameState::isHandOver() const {
    return current_stage_ == GameStage::SHOWDOWN || getPlayersInHand() <= 1;
}

std::vector<int> GameState::getWinners() const {
    std::vector<int> winners;
    
    // If only one player is left, they're the winner
    if (getPlayersInHand() == 1) {
        for (size_t i = 0; i < players_.size(); ++i) {
            if (players_[i].isActive()) {
                winners.push_back(i);
                break;
            }
        }
        return winners;
    }
    
    // If we're not at showdown yet, no winners
    if (current_stage_ != GameStage::SHOWDOWN) {
        return winners;
    }
    
    // Find best hand value among active players
    uint32_t best_value = 0;
    for (size_t i = 0; i < players_.size(); ++i) {
        if (!players_[i].hasFolded()) {
            uint32_t value = getHandValue(i);
            if (value > best_value) {
                best_value = value;
            }
        }
    }
    
    // Find all players with the best hand
    for (size_t i = 0; i < players_.size(); ++i) {
        if (!players_[i].hasFolded() && getHandValue(i) == best_value) {
            winners.push_back(i);
        }
    }
    
    return winners;
}

uint32_t GameState::getHandValue(int player_index) const {
    if (player_index < 0 || player_index >= static_cast<int>(players_.size())) {
        throw std::out_of_range("Player index out of range");
    }
    
    if (players_[player_index].hasFolded()) {
        return 0;
    }
    
    // Combine hole cards and community cards
    std::vector<Card> cards;
    cards.reserve(NUM_HOLE_CARDS + community_cards_.size());
    
    const auto& hole_cards = players_[player_index].getHoleCards();
    for (const auto& card : hole_cards) {
        cards.push_back(card);
    }
    
    for (const auto& card : community_cards_) {
        cards.push_back(card);
    }
    
    return hand_evaluator_.evaluate(cards);
}

std::string GameState::getHandDescription(int player_index) const {
    return hand_evaluator_.getHandDescription(getHandValue(player_index));
}

uint64_t GameState::getSeed() const {
    return seed_;
}

void GameState::setSeed(uint64_t seed) {
    seed_ = seed;
    rng_.seed(seed);
}

std::string GameState::toString() const {
    std::stringstream ss;
    
    // Game stage
    ss << "Stage: " << GAME_STAGE_STRINGS[static_cast<int>(current_stage_)] << "\n";
    
    // Pot
    ss << "Pot: " << formatChips(pot_) << "\n";
    
    // Community cards
    ss << "Board: ";
    if (community_cards_.empty()) {
        ss << "[]";
    } else {
        ss << "[";
        for (size_t i = 0; i < community_cards_.size(); ++i) {
            if (i > 0) ss << " ";
            ss << community_cards_[i];
        }
        ss << "]";
    }
    ss << "\n";
    
    // Players
    ss << "Players:\n";
    for (size_t i = 0; i < players_.size(); ++i) {
        const auto& player = players_[i];
        ss << (i == static_cast<size_t>(dealer_position_) ? "D " : "  ");
        ss << (i == static_cast<size_t>(current_player_index_) ? "-> " : "   ");
        ss << player.toString();
        
        if (player.getCurrentBet() > 0) {
            ss << " (bet: " << formatChips(player.getCurrentBet()) << ")";
        }
        
        if (current_stage_ == GameStage::SHOWDOWN && !player.hasFolded()) {
            ss << " - " << getHandDescription(i);
        }
        
        ss << "\n";
    }
    
    return ss.str();
}

void GameState::advanceToNextPlayer() {
    do {
        current_player_index_ = (current_player_index_ + 1) % players_.size();
    } while (!players_[current_player_index_].isActive() && !isHandOver());
}

void GameState::postBlinds() {
    // Determine small and big blind positions
    int small_blind_pos = (dealer_position_ + 1) % players_.size();
    int big_blind_pos = (dealer_position_ + 2) % players_.size();
    
    // Handle heads-up special case
    if (players_.size() == 2) {
        small_blind_pos = dealer_position_;
        big_blind_pos = (dealer_position_ + 1) % 2;
    }
    
    // Post small blind
    Player& small_blind_player = players_[small_blind_pos];
    int64_t sb_amount = std::min(small_blind_, small_blind_player.getStack());
    small_blind_player.adjustStack(-sb_amount);
    small_blind_player.setCurrentBet(sb_amount);
    
    // Post big blind
    Player& big_blind_player = players_[big_blind_pos];
    int64_t bb_amount = std::min(big_blind_, big_blind_player.getStack());
    big_blind_player.adjustStack(-bb_amount);
    big_blind_player.setCurrentBet(bb_amount);
    
    // Set initial bet level to big blind
    current_min_bet_ = bb_amount;
    min_raise_ = big_blind_;
    last_aggressor_ = big_blind_pos;
}

void GameState::distributePot() {
    // First, move all bets to the pot
    moveBetsToPot();
    
    // If only one player is left, they get the pot
    if (getPlayersInHand() == 1) {
        for (size_t i = 0; i < players_.size(); ++i) {
            if (players_[i].isActive()) {
                players_[i].adjustStack(pot_);
                pot_ = 0;
                return;
            }
        }
    }
    
    // Get winners
    auto winners = getWinners();
    
    // If no winners (should never happen), return
    if (winners.empty()) {
        return;
    }
    
    // Split the pot evenly among winners
    int64_t share = pot_ / winners.size();
    int64_t remainder = pot_ % winners.size();
    
    for (int winner : winners) {
        players_[winner].adjustStack(share);
    }
    
    // Give remainder to the first winner (closest to dealer)
    if (remainder > 0) {
        players_[winners[0]].adjustStack(remainder);
    }
    
    pot_ = 0;
}

bool GameState::isBettingRoundOver() const {
    // If only one player is left, betting is over
    if (getPlayersInHand() <= 1) {
        return true;
    }
    
    // If all players are all-in except possibly one, betting is over
    int active_not_all_in = 0;
    for (const auto& player : players_) {
        if (player.isActive() && !player.isAllIn()) {
            active_not_all_in++;
        }
    }
    
    if (active_not_all_in <= 1) {
        return true;
    }
    
    // Check if everyone has had a chance to act since the last aggression
    if (last_aggressor_ == -1) {
        return false;
    }
    
    // Check if all active players have the same bet amount
    int64_t target_bet = 0;
    bool first_active = true;
    
    for (const auto& player : players_) {
        if (player.isActive()) {
            if (first_active) {
                target_bet = player.getCurrentBet();
                first_active = false;
            } else if (player.getCurrentBet() != target_bet) {
                return false;
            }
        }
    }
    
    // Check if we've gone all the way around to the last aggressor
    return current_player_index_ == last_aggressor_;
}

int GameState::getActivePlayerCount() const {
    int count = 0;
    for (const auto& player : players_) {
        if (player.isActive()) {
            count++;
        }
    }
    return count;
}

int GameState::getPlayersInHand() const {
    int count = 0;
    for (const auto& player : players_) {
        if (!player.hasFolded()) {
            count++;
        }
    }
    return count;
}

void GameState::moveBetsToPot() {
    for (auto& player : players_) {
        pot_ += player.getCurrentBet();
        player.resetCurrentBet();
    }
}

} // namespace poker
