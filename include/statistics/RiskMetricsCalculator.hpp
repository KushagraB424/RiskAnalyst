#pragma once
#include "Types.hpp"
#include "Config.hpp"
#include <vector>

namespace RiskAnalyst {

class RiskMetricsCalculator {
public:
    static RiskMetrics calculate(const std::vector<double>& payoffs, const Config& config);
    
    // Abstraction for sorting so it can be swapped or compared with GPU sorting
    static void sortReturns(std::vector<double>& returns);
};

} // namespace RiskAnalyst
