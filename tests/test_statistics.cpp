#include <gtest/gtest.h>
#include "statistics/RiskMetricsCalculator.hpp"
#include "Config.hpp"
#include <vector>

using namespace RiskAnalyst;

TEST(RiskMetricsTest, CalculatesMetricsCorrectly) {
    Config config;
    // Dummy payoffs: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    // Total 11 elements
    std::vector<double> payoffs = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
    
    RiskMetrics metrics = RiskMetricsCalculator::calculate(payoffs, config);
    
    EXPECT_DOUBLE_EQ(metrics.meanReturn, 50.0);
    // Probability of loss: only 0.0 is <= 1e-8. So 1 / 11 = 0.090909...
    EXPECT_NEAR(metrics.probabilityOfLoss, 1.0 / 11.0, 1e-5);
    
    // VaR 95 (alpha=0.05). Index = 0.05 * 11 = 0
    EXPECT_DOUBLE_EQ(metrics.valueAtRisk95, 0.0);
    EXPECT_DOUBLE_EQ(metrics.expectedShortfall95, 0.0);
}
