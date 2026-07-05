#include "finance/OptionPricer.hpp"
#include <algorithm>
#include <stdexcept>

namespace RiskAnalyst {

std::vector<double> OptionPricer::calculatePayoffs(
    const std::vector<double>& finalPrices,
    const std::vector<double>& averagePrices,
    double strikePrice,
    OptionType type) {
    
    std::vector<double> payoffs(finalPrices.size());

    for (size_t i = 0; i < finalPrices.size(); ++i) {
        switch (type) {
            case OptionType::EuropeanCall:
                payoffs[i] = std::max(finalPrices[i] - strikePrice, 0.0);
                break;
            case OptionType::EuropeanPut:
                payoffs[i] = std::max(strikePrice - finalPrices[i], 0.0);
                break;
            case OptionType::AsianCall:
                payoffs[i] = std::max(averagePrices[i] - strikePrice, 0.0);
                break;
            default:
                throw std::invalid_argument("Unsupported option type.");
        }
    }

    return payoffs;
}

} // namespace RiskAnalyst
