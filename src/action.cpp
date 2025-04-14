#include "poker/action.h"
#include "poker/utils.h"
#include <sstream>
#include <stdexcept>

namespace poker {

Action::Action() : type_(ActionType::FOLD), amount_(0) {}

Action::Action(ActionType type) : type_(type), amount_(0) {
    if (type != ActionType::FOLD && type != ActionType::CHECK) {
        throw std::invalid_argument("This action type requires an amount");
    }
}

Action::Action(ActionType type, int64_t amount) : type_(type), amount_(amount) {
    if (type == ActionType::FOLD || type == ActionType::CHECK) {
        throw std::invalid_argument("This action type does not accept an amount");
    }
    
    if (amount < 0) {
        throw std::invalid_argument("Action amount cannot be negative");
    }
}

ActionType Action::getType() const {
    return type_;
}

int64_t Action::getAmount() const {
    return amount_;
}

Action Action::fold() {
    return Action(ActionType::FOLD);
}

Action Action::check() {
    return Action(ActionType::CHECK);
}

Action Action::call(int64_t amount) {
    return Action(ActionType::CALL, amount);
}

Action Action::bet(int64_t amount) {
    return Action(ActionType::BET, amount);
}

Action Action::raise(int64_t amount) {
    return Action(ActionType::RAISE, amount);
}

Action Action::allIn(int64_t amount) {
    return Action(ActionType::ALL_IN, amount);
}

std::string Action::toString() const {
    std::stringstream ss;
    ss << ACTION_TYPE_STRINGS[static_cast<int>(type_)];
    
    if (type_ == ActionType::CALL || type_ == ActionType::BET || 
        type_ == ActionType::RAISE || type_ == ActionType::ALL_IN) {
        ss << " " << formatChips(amount_);
    }
    
    return ss.str();
}

} // namespace poker
