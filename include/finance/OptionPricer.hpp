#pragma once
#include "Types.hpp"
#include <vector>

namespace RiskAnalyst {

class OptionPricer {
public:
    /**
     * @brief Computes the payoffs for a given set of simulated price paths.
     * @param finalPrices The terminal stock prices for each path.
     * @param averagePrices The arithmetic average of stock prices for each path (used for Asian options).
     * @param strikePrice The strike price of the option.
     * @param type The type of option to price.
     * @return std::vector<double> The computed payoffs before discounting.
     */
    static std::vector<double> calculatePayoffs(
        const std::vector<double>& finalPrices,
        const std::vector<double>& averagePrices,
        double strikePrice,
        OptionType type);
};

} // namespace RiskAnalyst
