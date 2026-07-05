#pragma once
#include "MonteCarloEngine.hpp"

namespace RiskAnalyst {

class CpuOpenMPEngine : public MonteCarloEngine {
public:
    SimulationResult runSimulation(const Config& config) override;
};

} // namespace RiskAnalyst
