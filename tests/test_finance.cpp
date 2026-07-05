#include <gtest/gtest.h>
#include "finance/CpuSequentialEngine.hpp"
#include "finance/CpuOpenMPEngine.hpp"
#include "Config.hpp"
#include <cmath>

using namespace RiskAnalyst;

class FinanceTest : public ::testing::Test {
protected:
    Config config;
    
    void SetUp() override {
        config.market.spotPrice = 100.0;
        config.market.strikePrice = 100.0;
        config.market.riskFreeRate = 0.05;
        config.market.volatility = 0.20;
        config.market.timeToMaturity = 1.0;
        
        config.simulation.numPaths = 10000;
        config.simulation.numSteps = 252;
        config.simulation.seed = 42;
    }
};

TEST_F(FinanceTest, EuropeanCallPricesMatchBlackScholesApprox) {
    config.simulation.optionType = OptionType::EuropeanCall;
    
    CpuSequentialEngine seqEngine;
    SimulationResult seqResult = seqEngine.runSimulation(config);
    
    CpuOpenMPEngine ompEngine;
    SimulationResult ompResult = ompEngine.runSimulation(config);
    
    // Calculate average discounted payoff (simulated option price)
    double seqPrice = 0.0;
    for (double p : seqResult.payoffs) seqPrice += p;
    seqPrice /= seqResult.payoffs.size();
    
    double ompPrice = 0.0;
    for (double p : ompResult.payoffs) ompPrice += p;
    ompPrice /= ompResult.payoffs.size();
    
    // Black-Scholes price for these parameters is approx 10.45
    EXPECT_NEAR(seqPrice, 10.45, 0.5);
    // Note: OMP result might differ slightly from sequential due to different random sequences per thread
    EXPECT_NEAR(ompPrice, 10.45, 0.5);
}

TEST_F(FinanceTest, EuropeanPutPricesMatchBlackScholesApprox) {
    config.simulation.optionType = OptionType::EuropeanPut;
    
    CpuSequentialEngine seqEngine;
    SimulationResult result = seqEngine.runSimulation(config);
    
    double price = 0.0;
    for (double p : result.payoffs) price += p;
    price /= result.payoffs.size();
    
    // Put-Call Parity: P = C + K*e^(-rT) - S0
    // P = 10.45 + 100*exp(-0.05) - 100 = 10.45 + 95.12 - 100 = 5.57
    EXPECT_NEAR(price, 5.57, 0.5);
}

TEST_F(FinanceTest, AsianCallIsCheaperThanEuropeanCall) {
    config.simulation.optionType = OptionType::AsianCall;
    
    CpuSequentialEngine seqEngine;
    SimulationResult asianResult = seqEngine.runSimulation(config);
    
    double asianPrice = 0.0;
    for (double p : asianResult.payoffs) asianPrice += p;
    asianPrice /= asianResult.payoffs.size();
    
    // Asian options are typically cheaper than European due to averaging reducing volatility
    EXPECT_LT(asianPrice, 10.45);
    EXPECT_GT(asianPrice, 0.0);
}
