#pragma once

#include "poker/constants.h"
#include <string>

namespace poker {

/**
 * Represents a poker action (fold, check, call, bet, raise, all-in).
 */
class Action {
public:
    // Default constructor
    Action();
    
    // Create an action with no amount (fold, check)
    explicit Action(ActionType type);
    
    // Create an action with an amount (call, bet, raise, all-in)
    Action(ActionType type, int64_t amount);
    
    // Get the action type
    ActionType getType() const;
    
    // Get the action amount (only relevant for call, bet, raise, all-in)
    int64_t getAmount() const;
    
    // Create a fold action
    static Action fold();
    
    // Create a check action
    static Action check();
    
    // Create a call action
    static Action call(int64_t amount);
    
    // Create a bet action
    static Action bet(int64_t amount);
    
    // Create a raise action
    static Action raise(int64_t amount);
    
    // Create an all-in action
    static Action allIn(int64_t amount);
    
    // String representation
    std::string toString() const;

private:
    ActionType type_;
    int64_t amount_ = 0;
};

} // namespace poker
