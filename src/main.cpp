#include "Config.hpp"
#include "benchmark/GpuDetector.hpp"
#include "benchmark/BenchmarkPipeline.hpp"
#include <iostream>
#include <filesystem>
#include <omp.h>

using namespace RiskAnalyst;

int main() {
    std::filesystem::path outputDir = "results";
    std::filesystem::create_directories(outputDir);
    
    Config config;
    auto gpuInfo = GpuDetector::getGpuInfo();
    
    std::cout << "===========================================\n";
    std::cout << "Risk Analyst Benchmark Configuration\n";
    std::cout << "===========================================\n\n";
    std::cout << "Random Seed: " << config.simulation.seed << "\n";
    std::cout << "Option Type: European Call\n";
    std::cout << "Spot Price: " << config.market.spotPrice << "\n";
    std::cout << "Strike Price: " << config.market.strikePrice << "\n";
    std::cout << "Volatility: " << config.market.volatility * 100 << "%\n";
    std::cout << "Risk-Free Rate: " << config.market.riskFreeRate * 100 << "%\n";
    std::cout << "Time to Maturity: " << config.market.timeToMaturity << " Year(s)\n\n";
    
    std::cout << "Simulation Counts:\n";
    for (size_t c : config.benchmark.pathCounts) {
        if (c >= 1000000) std::cout << c / 1000000 << "M\n";
        else std::cout << c / 1000 << "K\n";
    }
    
    #ifdef _OPENMP
    std::cout << "\nOpenMP Threads: " << omp_get_max_threads() << "\n\n";
    #else
    std::cout << "\nOpenMP Threads: 1 (OpenMP not found)\n\n";
    #endif
    
    std::cout << "GPU:\n";
    std::cout << "NVIDIA " << gpuInfo.name << "\n";
    std::cout << "CUDA " << gpuInfo.cudaVersion << "\n";
    std::cout << "===========================================\n\n";

    GpuDetector::writeHardwareInfo((outputDir / "hardware_info.txt").string());
    
    BenchmarkPipeline pipeline(config, outputDir.string());
    pipeline.runAll();
    pipeline.saveResultsCsv((outputDir / "benchmark_results.csv").string());
    
    std::cout << "\n===========================================\n";
    std::cout << "Benchmark Complete\n\n";
    std::cout << "Generated Files:\n\n";
    std::cout << outputDir.string() << "/\n";
    std::cout << "|-- benchmark_results.csv\n";
    std::cout << "|-- hardware_info.txt\n";
    std::cout << "|-- benchmark_config.txt\n";
    if (std::filesystem::exists(outputDir / "payoffs_sample.csv")) {
        std::cout << "|-- payoffs_sample.csv\n";
    }
    std::cout << "\nNext Step:\n";
    std::cout << "Run visualization/generate_plots.py to generate the figures and benchmark summary.\n";
    std::cout << "===========================================\n";
    
    return 0;
}
