#pragma once
#include "Config.hpp"
#include "Types.hpp"

namespace RiskAnalyst {

class MonteCarloEngine {
public:
    virtual ~MonteCarloEngine() = default;
    virtual SimulationResult runSimulation(const Config& config) = 0;
};

} // namespace RiskAnalyst
