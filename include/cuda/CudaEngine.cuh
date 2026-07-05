#pragma once
#include "finance/MonteCarloEngine.hpp"

namespace RiskAnalyst {

class CudaEngine : public MonteCarloEngine {
public:
    SimulationResult runSimulation(const Config& config) override;
};

} // namespace RiskAnalyst
