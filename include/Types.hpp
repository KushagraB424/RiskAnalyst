#pragma once
#include <vector>
#include <string>

namespace RiskAnalyst {

enum class OptionType {
    EuropeanCall,
    EuropeanPut,
    AsianCall
};

struct MarketParameters {
    double spotPrice = 100.0;
    double strikePrice = 100.0;
    double riskFreeRate = 0.05;
    double volatility = 0.20;
    double timeToMaturity = 1.0;
};

struct SimulationParameters {
    size_t numPaths = 10000;
    size_t numSteps = 252;
    OptionType optionType = OptionType::EuropeanCall;
    unsigned int seed = 42; 
};

struct BenchmarkParameters {
    std::vector<size_t> pathCounts = {10000, 100000, 1000000, 10000000, 50000000};
    std::vector<double> confidenceLevels = {0.95, 0.99};
    unsigned int seed = 42;
};

struct SimulationResult {
    std::vector<double> finalPrices;
    std::vector<double> payoffs;
    
    size_t totalSimulations = 0;
    size_t timeSteps = 0;
    OptionType optionType = OptionType::EuropeanCall;
    
    double hostToDeviceTimeMs = 0.0;
    double kernelTimeMs = 0.0;
    double deviceToHostTimeMs = 0.0;
    double totalExecutionTimeMs = 0.0;
    double throughput = 0.0;
    
    double gpuSum = 0.0; 
    
    // Hardware utilization metrics
    int threadsUsed = 1; // CPU OpenMP threads
    int blockSize = 0;   // CUDA block size
    int gridSize = 0;    // CUDA grid size
};

struct RiskMetrics {
    double valueAtRisk95;
    double valueAtRisk99;
    double expectedShortfall95;
    double expectedShortfall99;
    double meanReturn;
    double variance;
    double standardDeviation;
    double sharpeRatio;
    double probabilityOfLoss;
};

struct GpuInfo {
    std::string name;
    int cudaVersion;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    int clockRateKHz;
    size_t totalGlobalMemBytes;
    int multiProcessorCount;
    size_t sharedMemPerBlock;
    int warpSize;
};

struct BenchmarkResult {
    std::string engineName;
    size_t numPaths;
    double h2dTimeMs;
    double kernelTimeMs;
    double d2hTimeMs;
    double totalTimeMs;
    double throughput; 
    double speedupVsSequential;
    double speedupVsOpenMP;
    size_t peakMemoryBytes;
    
    int threadsUsed;
    int blockSize;
    int gridSize;
    
    double meanReturn;
    double standardDeviation;
    double var95;
    double cvar95;
    double relativeError; 
};

} // namespace RiskAnalyst
