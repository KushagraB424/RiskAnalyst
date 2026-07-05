#pragma once
#include "Types.hpp"

namespace RiskAnalyst {

struct Config {
    MarketParameters market;
    SimulationParameters simulation;
    BenchmarkParameters benchmark;
};

} // namespace RiskAnalyst
