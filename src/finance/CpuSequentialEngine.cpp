#include "finance/CpuSequentialEngine.hpp"
#include "finance/OptionPricer.hpp"
#include <cmath>
#include <random>
#include <chrono>

namespace RiskAnalyst {

/**
 * @brief Runs a Monte Carlo simulation on a single CPU thread.
 * Generates asset price paths using Geometric Brownian Motion.
 * 
 * @note Time Complexity: O(N * S) where N is numPaths and S is numSteps.
 * @note Memory Complexity: O(N) to store final prices and average prices.
 */
SimulationResult CpuSequentialEngine::runSimulation(const Config& config) {
    auto start = std::chrono::high_resolution_clock::now();
    
    const size_t numPaths = config.simulation.numPaths;
    const size_t numSteps = config.simulation.numSteps;
    const double S0 = config.market.spotPrice;
    const double r = config.market.riskFreeRate;
    const double v = config.market.volatility;
    const double T = config.market.timeToMaturity;
    
    const double dt = T / static_cast<double>(numSteps);
    const double drift = (r - 0.5 * v * v) * dt;
    const double vol = v * std::sqrt(dt);

    std::mt19937_64 rng(config.simulation.seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    SimulationResult result;
    result.finalPrices.resize(numPaths);
    std::vector<double> averagePrices(numPaths, 0.0);

    for (size_t i = 0; i < numPaths; ++i) {
        double currentPrice = S0;
        double sumPrices = 0.0;
        
        for (size_t step = 0; step < numSteps; ++step) {
            double Z = norm(rng);
            currentPrice = currentPrice * std::exp(drift + vol * Z);
            sumPrices += currentPrice;
        }
        
        result.finalPrices[i] = currentPrice;
        averagePrices[i] = sumPrices / static_cast<double>(numSteps);
    }

    result.payoffs = OptionPricer::calculatePayoffs(
        result.finalPrices, averagePrices, config.market.strikePrice, config.simulation.optionType);

    const double discountFactor = std::exp(-r * T);
    for (size_t i = 0; i < numPaths; ++i) {
        result.payoffs[i] *= discountFactor;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    result.totalSimulations = numPaths;
    result.timeSteps = numSteps;
    result.optionType = config.simulation.optionType;
    result.totalExecutionTimeMs = elapsed.count();
    result.throughput = (numPaths / (result.totalExecutionTimeMs / 1000.0));

    return result;
}

} // namespace RiskAnalyst
