#include "Config.hpp"
#include "benchmark/GpuDetector.hpp"
#include "benchmark/BenchmarkPipeline.hpp"
#include <iostream>
#include <filesystem>

using namespace RiskAnalyst;

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "Risk Analyst: GPU Accelerated Financial Risk Analytics" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    std::filesystem::create_directories("../results");
    
    GpuDetector::writeHardwareInfo("../results/hardware_info.txt");
    
    Config config;
    BenchmarkPipeline pipeline(config);
    pipeline.runAll();
    pipeline.saveResultsCsv("../results/benchmark_results.csv");
    
    std::cout << "Execution complete." << std::endl;
    return 0;
}
