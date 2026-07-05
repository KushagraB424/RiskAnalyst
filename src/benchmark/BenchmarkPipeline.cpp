#include "benchmark/BenchmarkPipeline.hpp"
#include "finance/CpuSequentialEngine.hpp"
#include "finance/CpuOpenMPEngine.hpp"
#include "cuda/CudaEngine.cuh"
#include "cuda/CudaEngineOptimized.cuh"
#include "statistics/RiskMetricsCalculator.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace RiskAnalyst {

BenchmarkPipeline::BenchmarkPipeline(const Config& config, const std::string& outputDir) : m_config(config), m_outputDir(outputDir) {}

void BenchmarkPipeline::savePayoffsSample(const std::vector<double>& payoffs, const std::string& filename, size_t maxSamples) {
    std::ofstream out(filename);
    out << "Payoff\n";
    size_t count = std::min(payoffs.size(), maxSamples);
    for (size_t i = 0; i < count; ++i) {
        out << payoffs[i] << "\n";
    }
}

void BenchmarkPipeline::saveConfig(const std::string& filename) {
    std::ofstream out(filename);
    out << "Random Seed: " << m_config.simulation.seed << "\n";
    out << "Time Steps: " << m_config.simulation.numSteps << "\n";
    out << "Option Type: European Call\n";
    out << "Spot Price (S0): " << m_config.market.spotPrice << "\n";
    out << "Strike Price (K): " << m_config.market.strikePrice << "\n";
    out << "Risk-Free Rate (r): " << m_config.market.riskFreeRate << "\n";
    out << "Volatility (v): " << m_config.market.volatility << "\n";
    out << "Time to Maturity (T): " << m_config.market.timeToMaturity << " years\n";
}

void BenchmarkPipeline::runAll() {
    std::cout << "Running Benchmark Pipeline..." << std::endl;
    saveConfig(m_outputDir + "/benchmark_config.txt");
    
    CpuSequentialEngine cpuSeq;
    CpuOpenMPEngine cpuOmp;
    CudaEngine cudaBase;
    CudaEngineOptimized cudaOpt;
    
    for (size_t paths : m_config.benchmark.pathCounts) {
        std::cout << "\n[ Benchmarking N = " << paths << " ]" << std::endl;
        Config iterConfig = m_config;
        iterConfig.simulation.numPaths = paths;
        
        auto resSeq = cpuSeq.runSimulation(iterConfig);
        auto metricsSeq = RiskMetricsCalculator::calculate(resSeq.payoffs, iterConfig);
        double refMean = metricsSeq.meanReturn;
        
        m_results.push_back({"CPU Sequential", paths, 0, 0, 0, 
                             resSeq.totalExecutionTimeMs, resSeq.throughput, 
                             1.0, 0.0, 0, 
                             resSeq.threadsUsed, resSeq.blockSize, resSeq.gridSize,
                             metricsSeq.meanReturn, metricsSeq.standardDeviation, metricsSeq.valueAtRisk95, metricsSeq.expectedShortfall95, 0.0});
                             
        auto resOmp = cpuOmp.runSimulation(iterConfig);
        auto metricsOmp = RiskMetricsCalculator::calculate(resOmp.payoffs, iterConfig);
        m_results.push_back({"CPU OpenMP", paths, 0, 0, 0, 
                             resOmp.totalExecutionTimeMs, resOmp.throughput, 
                             resSeq.totalExecutionTimeMs / resOmp.totalExecutionTimeMs, 1.0, 0, 
                             resOmp.threadsUsed, resOmp.blockSize, resOmp.gridSize,
                             metricsOmp.meanReturn, metricsOmp.standardDeviation, metricsOmp.valueAtRisk95, metricsOmp.expectedShortfall95, 
                             std::abs(metricsOmp.meanReturn - refMean)/refMean});
                             
        auto resBase = cudaBase.runSimulation(iterConfig);
        auto metricsBase = RiskMetricsCalculator::calculate(resBase.payoffs, iterConfig);
        m_results.push_back({"CUDA Baseline", paths, 0, 0, 0, 
                             resBase.totalExecutionTimeMs, resBase.throughput, 
                             resSeq.totalExecutionTimeMs / resBase.totalExecutionTimeMs, 
                             resOmp.totalExecutionTimeMs / resBase.totalExecutionTimeMs, 0, 
                             resBase.threadsUsed, resBase.blockSize, resBase.gridSize,
                             metricsBase.meanReturn, metricsBase.standardDeviation, metricsBase.valueAtRisk95, metricsBase.expectedShortfall95, 
                             std::abs(metricsBase.meanReturn - refMean)/refMean});
                             
        auto resOpt = cudaOpt.runSimulation(iterConfig);
        auto metricsOpt = RiskMetricsCalculator::calculate(resOpt.payoffs, iterConfig);
        m_results.push_back({"CUDA Optimized", paths, resOpt.hostToDeviceTimeMs, resOpt.kernelTimeMs, resOpt.deviceToHostTimeMs, 
                             resOpt.totalExecutionTimeMs, resOpt.throughput, 
                             resSeq.totalExecutionTimeMs / resOpt.totalExecutionTimeMs, 
                             resOmp.totalExecutionTimeMs / resOpt.totalExecutionTimeMs, 0, 
                             resOpt.threadsUsed, resOpt.blockSize, resOpt.gridSize,
                             metricsOpt.meanReturn, metricsOpt.standardDeviation, metricsOpt.valueAtRisk95, metricsOpt.expectedShortfall95, 
                             std::abs(metricsOpt.meanReturn - refMean)/refMean});
                             
        if (paths == m_config.benchmark.pathCounts.back()) {
            savePayoffsSample(resOpt.payoffs, m_outputDir + "/payoffs_sample.csv");
        }
    }
}

void BenchmarkPipeline::saveResultsCsv(const std::string& filename) {
    std::ofstream out(filename);
    out << "Engine,NumPaths,H2DTime,KernelTime,D2HTime,TotalTime,Throughput,SpeedupSeq,SpeedupOmp,ThreadsUsed,BlockSize,GridSize,Mean,StdDev,VaR95,CVaR95,RelError\n";
    for (const auto& res : m_results) {
        out << res.engineName << "," << res.numPaths << "," 
            << res.h2dTimeMs << "," << res.kernelTimeMs << "," << res.d2hTimeMs << "," 
            << res.totalTimeMs << "," << res.throughput << "," 
            << res.speedupVsSequential << "," << res.speedupVsOpenMP << ","
            << res.threadsUsed << "," << res.blockSize << "," << res.gridSize << ","
            << res.meanReturn << "," << res.standardDeviation << "," << res.var95 << "," << res.cvar95 << "," << res.relativeError << "\n";
    }
    std::cout << "Results saved to " << filename << std::endl;
}

} // namespace RiskAnalyst
