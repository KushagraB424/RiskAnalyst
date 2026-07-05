#include "statistics/RiskMetricsCalculator.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace RiskAnalyst {

void RiskMetricsCalculator::sortReturns(std::vector<double>& returns) {
    std::sort(returns.begin(), returns.end());
}

/**
 * @brief Computes various risk metrics from an array of discounted payoffs (P&L).
 * @param payoffs A vector of discounted payoffs from the simulation.
 * @param config Simulation and market configuration, providing confidence levels.
 * @return RiskMetrics Struct containing Mean, Variance, StdDev, VaR, CVaR, etc.
 * 
 * @note Time Complexity: O(N log N) dominated by the sorting step for VaR/CVaR.
 * @note Memory Complexity: O(N) to store the sorted copy of the returns array.
 */
RiskMetrics RiskMetricsCalculator::calculate(const std::vector<double>& payoffs, const Config& config) {
    RiskMetrics metrics = {};
    if (payoffs.empty()) return metrics;

    size_t n = payoffs.size();
    
    // O(N) operations
    double sum = std::accumulate(payoffs.begin(), payoffs.end(), 0.0);
    metrics.meanReturn = sum / static_cast<double>(n);

    double sqSum = 0.0;
    for (double p : payoffs) {
        double diff = p - metrics.meanReturn;
        sqSum += diff * diff;
    }
    metrics.variance = sqSum / static_cast<double>(n - 1);
    metrics.standardDeviation = std::sqrt(metrics.variance);

    if (metrics.standardDeviation > 1e-8) {
        metrics.sharpeRatio = metrics.meanReturn / metrics.standardDeviation;
    } else {
        metrics.sharpeRatio = 0.0;
    }

    // O(N log N) sorting step
    std::vector<double> sortedPayoffs = payoffs;
    sortReturns(sortedPayoffs);

    double alpha95 = 1.0 - 0.95;
    size_t index95 = static_cast<size_t>(alpha95 * n);
    metrics.valueAtRisk95 = sortedPayoffs[std::min(index95, n - 1)];

    double alpha99 = 1.0 - 0.99;
    size_t index99 = static_cast<size_t>(alpha99 * n);
    metrics.valueAtRisk99 = sortedPayoffs[std::min(index99, n - 1)];

    auto calcCVaR = [&](size_t idx) {
        if (idx == 0) return sortedPayoffs[0];
        double tailSum = 0.0;
        for (size_t i = 0; i < idx; ++i) {
            tailSum += sortedPayoffs[i];
        }
        return tailSum / static_cast<double>(idx);
    };

    metrics.expectedShortfall95 = calcCVaR(index95);
    metrics.expectedShortfall99 = calcCVaR(index99);

    size_t lossCount = 0;
    for (double p : sortedPayoffs) {
        if (p <= 1e-8) {
            lossCount++;
        } else {
            break;
        }
    }
    metrics.probabilityOfLoss = static_cast<double>(lossCount) / static_cast<double>(n);

    return metrics;
}

} // namespace RiskAnalyst
