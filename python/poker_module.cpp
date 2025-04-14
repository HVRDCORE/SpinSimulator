#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "poker/card.h"
#include "poker/deck.h"
#include "poker/hand_evaluator.h"
#include "poker/player.h"
#include "poker/action.h"
#include "poker/game_state.h"
#include "poker/spingo_game.h"

namespace py = pybind11;

PYBIND11_MODULE(poker_core, m) {
    m.doc() = "C++ Spin & Go poker simulator with Python bindings";
    
    // Enums
    py::enum_<poker::Suit>(m, "Suit")
        .value("CLUBS", poker::Suit::CLUBS)
        .value("DIAMONDS", poker::Suit::DIAMONDS)
        .value("HEARTS", poker::Suit::HEARTS)
        .value("SPADES", poker::Suit::SPADES)
        .export_values();
    
    py::enum_<poker::Rank>(m, "Rank")
        .value("TWO", poker::Rank::TWO)
        .value("THREE", poker::Rank::THREE)
        .value("FOUR", poker::Rank::FOUR)
        .value("FIVE", poker::Rank::FIVE)
        .value("SIX", poker::Rank::SIX)
        .value("SEVEN", poker::Rank::SEVEN)
        .value("EIGHT", poker::Rank::EIGHT)
        .value("NINE", poker::Rank::NINE)
        .value("TEN", poker::Rank::TEN)
        .value("JACK", poker::Rank::JACK)
        .value("QUEEN", poker::Rank::QUEEN)
        .value("KING", poker::Rank::KING)
        .value("ACE", poker::Rank::ACE)
        .export_values();
    
    py::enum_<poker::HandType>(m, "HandType")
        .value("HIGH_CARD", poker::HandType::HIGH_CARD)
        .value("PAIR", poker::HandType::PAIR)
        .value("TWO_PAIR", poker::HandType::TWO_PAIR)
        .value("THREE_OF_A_KIND", poker::HandType::THREE_OF_A_KIND)
        .value("STRAIGHT", poker::HandType::STRAIGHT)
        .value("FLUSH", poker::HandType::FLUSH)
        .value("FULL_HOUSE", poker::HandType::FULL_HOUSE)
        .value("FOUR_OF_A_KIND", poker::HandType::FOUR_OF_A_KIND)
        .value("STRAIGHT_FLUSH", poker::HandType::STRAIGHT_FLUSH)
        .value("ROYAL_FLUSH", poker::HandType::ROYAL_FLUSH)
        .export_values();
    
    py::enum_<poker::GameStage>(m, "GameStage")
        .value("PREFLOP", poker::GameStage::PREFLOP)
        .value("FLOP", poker::GameStage::FLOP)
        .value("TURN", poker::GameStage::TURN)
        .value("RIVER", poker::GameStage::RIVER)
        .value("SHOWDOWN", poker::GameStage::SHOWDOWN)
        .export_values();
    
    py::enum_<poker::ActionType>(m, "ActionType")
        .value("FOLD", poker::ActionType::FOLD)
        .value("CHECK", poker::ActionType::CHECK)
        .value("CALL", poker::ActionType::CALL)
        .value("BET", poker::ActionType::BET)
        .value("RAISE", poker::ActionType::RAISE)
        .value("ALL_IN", poker::ActionType::ALL_IN)
        .export_values();
    
    // Card class
    py::class_<poker::Card>(m, "Card")
        .def(py::init<>())
        .def(py::init<poker::Rank, poker::Suit>())
        .def(py::init<int>())
        .def(py::init<const std::string&>())
        .def("get_rank", &poker::Card::getRank)
        .def("get_suit", &poker::Card::getSuit)
        .def("get_id", &poker::Card::getId)
        .def("to_string", &poker::Card::toString)
        .def("is_valid", &poker::Card::isValid)
        .def("__eq__", &poker::Card::operator==)
        .def("__ne__", &poker::Card::operator!=)
        .def("__repr__", &poker::Card::toString);
    
    // Deck class
    py::class_<poker::Deck>(m, "Deck")
        .def(py::init<>())
        .def(py::init<const std::vector<poker::Card>&>())
        .def("shuffle", &poker::Deck::shuffle)
        // Add a convenience method for Python that takes a seed directly
        .def("shuffle", [](poker::Deck& d, uint32_t seed) {
            std::mt19937 rng(seed);
            d.shuffle(rng);
        })
        .def("deal_card", &poker::Deck::dealCard)
        .def("reset", &poker::Deck::reset)
        .def("cards_remaining", &poker::Deck::cardsRemaining)
        .def("get_cards", &poker::Deck::getCards)
        .def("remove_card", &poker::Deck::removeCard);
    
    // HandEvaluator class
    py::class_<poker::HandEvaluator>(m, "HandEvaluator")
        .def(py::init<>())
        .def("evaluate", py::overload_cast<const std::vector<poker::Card>&>(&poker::HandEvaluator::evaluate, py::const_))
        .def("evaluate", py::overload_cast<const std::vector<poker::Card>&, const std::vector<poker::Card>&>(&poker::HandEvaluator::evaluate, py::const_))
        .def("get_hand_type", &poker::HandEvaluator::getHandType)
        .def("get_hand_description", &poker::HandEvaluator::getHandDescription)
        .def("find_best_hand", &poker::HandEvaluator::findBestHand);
    
    // Player class
    py::class_<poker::Player>(m, "Player")
        .def(py::init<int, int64_t, const std::string&>(), 
             py::arg("id"), py::arg("initial_stack"), py::arg("name") = "")
        .def("get_id", &poker::Player::getId)
        .def("get_name", &poker::Player::getName)
        .def("set_hole_cards", &poker::Player::setHoleCards)
        .def("get_hole_cards", &poker::Player::getHoleCards)
        .def("get_stack", &poker::Player::getStack)
        .def("adjust_stack", &poker::Player::adjustStack)
        .def("get_current_bet", &poker::Player::getCurrentBet)
        .def("set_current_bet", &poker::Player::setCurrentBet)
        .def("reset_current_bet", &poker::Player::resetCurrentBet)
        .def("is_all_in", &poker::Player::isAllIn)
        .def("has_folded", &poker::Player::hasFolded)
        .def("set_folded", &poker::Player::setFolded)
        .def("is_active", &poker::Player::isActive)
        .def("reset_for_new_hand", &poker::Player::resetForNewHand)
        .def("to_string", &poker::Player::toString)
        .def("__repr__", &poker::Player::toString);
    
    // Action class
    py::class_<poker::Action>(m, "Action")
        .def(py::init<>())
        .def(py::init<poker::ActionType>())
        .def(py::init<poker::ActionType, int64_t>())
        .def("get_type", &poker::Action::getType)
        .def("get_amount", &poker::Action::getAmount)
        .def_static("fold", &poker::Action::fold)
        .def_static("check", &poker::Action::check)
        .def_static("call", &poker::Action::call)
        .def_static("bet", &poker::Action::bet)
        .def_static("raise", &poker::Action::raise)
        .def_static("all_in", &poker::Action::allIn)
        .def("to_string", &poker::Action::toString)
        .def("__repr__", &poker::Action::toString);
    
    // GameState class
    py::class_<poker::GameState>(m, "GameState")
        .def(py::init<int, int64_t, int64_t, int64_t>(),
             py::arg("num_players"), py::arg("initial_stack"), 
             py::arg("small_blind"), py::arg("big_blind"))
        .def("reset_for_new_hand", &poker::GameState::resetForNewHand)
        .def("deal_hole_cards", &poker::GameState::dealHoleCards)
        .def("deal_flop", &poker::GameState::dealFlop)
        .def("deal_turn", &poker::GameState::dealTurn)
        .def("deal_river", &poker::GameState::dealRiver)
        .def("advance_stage", &poker::GameState::advanceStage)
        .def("get_current_stage", &poker::GameState::getCurrentStage)
        .def("get_community_cards", &poker::GameState::getCommunityCards)
        .def("get_players", &poker::GameState::getPlayers, py::return_value_policy::reference)
        .def("get_pot", &poker::GameState::getPot)
        .def("get_current_player_index", &poker::GameState::getCurrentPlayerIndex)
        .def("get_dealer_position", &poker::GameState::getDealerPosition)
        .def("get_small_blind", &poker::GameState::getSmallBlind)
        .def("get_big_blind", &poker::GameState::getBigBlind)
        .def("get_current_min_bet", &poker::GameState::getCurrentMinBet)
        .def("get_min_raise", &poker::GameState::getMinRaise)
        .def("apply_action", &poker::GameState::applyAction)
        .def("get_legal_actions", &poker::GameState::getLegalActions)
        .def("is_hand_over", &poker::GameState::isHandOver)
        .def("get_winners", &poker::GameState::getWinners)
        .def("get_hand_value", &poker::GameState::getHandValue)
        .def("get_hand_description", &poker::GameState::getHandDescription)
        .def("get_seed", &poker::GameState::getSeed)
        .def("set_seed", &poker::GameState::setSeed)
        // Add a convenience version that takes an integer seed
        .def("set_seed", [](poker::GameState& gs, uint32_t seed) {
            gs.setSeed(seed);
        })
        .def("to_string", &poker::GameState::toString)
        .def("__repr__", &poker::GameState::toString);
    
    // SpinGoGame class
    py::class_<poker::SpinGoGame>(m, "SpinGoGame")
        .def(py::init<int, int64_t, int64_t, int64_t, float>(),
             py::arg("num_players") = 3, 
             py::arg("buy_in") = 500, 
             py::arg("small_blind") = 10, 
             py::arg("big_blind") = 20,
             py::arg("prize_multiplier") = 2.0f)
        .def("play", &poker::SpinGoGame::play)
        .def("play_hand", &poker::SpinGoGame::playHand)
        .def("play_to_completion", &poker::SpinGoGame::playToCompletion)
        .def("get_game_state", &poker::SpinGoGame::getGameState, py::return_value_policy::reference)
        .def("is_tournament_over", &poker::SpinGoGame::isTournamentOver)
        .def("get_tournament_winner", &poker::SpinGoGame::getTournamentWinner)
        .def("get_prize_pool", &poker::SpinGoGame::getPrizePool)
        .def("set_callback", &poker::SpinGoGame::setCallback)
        .def("set_seed", &poker::SpinGoGame::setSeed)
        // Add a convenience version that takes an integer seed
        .def("set_seed", [](poker::SpinGoGame& game, uint32_t seed) {
            game.setSeed(seed);
        })
        .def("to_string", &poker::SpinGoGame::toString)
        .def("__repr__", &poker::SpinGoGame::toString);
}
