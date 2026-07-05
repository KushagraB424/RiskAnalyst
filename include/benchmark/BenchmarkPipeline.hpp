#pragma once
#include "Config.hpp"
#include "Types.hpp"
#include <string>
#include <vector>

namespace RiskAnalyst {

class BenchmarkPipeline {
public:
    BenchmarkPipeline(const Config& config);
    void runAll();
    void saveResultsCsv(const std::string& filename);
    void savePayoffsSample(const std::vector<double>& payoffs, const std::string& filename, size_t maxSamples = 100000);
    
private:
    Config m_config;
    std::vector<BenchmarkResult> m_results;
};

} // namespace RiskAnalyst
