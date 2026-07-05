#include <gtest/gtest.h>
#include "finance/CpuSequentialEngine.hpp"
#include "cuda/CudaEngine.cuh"
#include "cuda/CudaEngineOptimized.cuh"
#include "Config.hpp"
#include <cmath>

using namespace RiskAnalyst;

class CudaValidationTest : public ::testing::Test {
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

TEST_F(CudaValidationTest, OptimizedCudaMatchesCpuWithinTolerance) {
    config.simulation.optionType = OptionType::EuropeanCall;
    
    CpuSequentialEngine cpuEngine;
    SimulationResult cpuResult = cpuEngine.runSimulation(config);
    
    CudaEngineOptimized cudaOptEngine;
    SimulationResult gpuResult = cudaOptEngine.runSimulation(config);
    
    double cpuMean = 0.0;
    for(double p : cpuResult.payoffs) cpuMean += p;
    cpuMean /= cpuResult.payoffs.size();
    
    double gpuMean = gpuResult.gpuSum / gpuResult.payoffs.size();
    
    double absError = std::abs(cpuMean - gpuMean);
    double relError = absError / cpuMean;
    
    EXPECT_LT(relError, 0.02); // 2% tolerance
    
    std::cout << "[          ] CPU Mean: " << cpuMean << ", GPU Optimized Mean: " << gpuMean << std::endl;
    std::cout << "[          ] Abs Error: " << absError << ", Rel Error: " << relError * 100.0 << "%" << std::endl;
    std::cout << "[          ] GPU Kernel Time: " << gpuResult.kernelTimeMs << " ms" << std::endl;
}
