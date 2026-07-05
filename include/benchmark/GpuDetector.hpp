#pragma once
#include "Types.hpp"
#include <string>

namespace RiskAnalyst {

class GpuDetector {
public:
    static GpuInfo getGpuInfo();
    static void writeHardwareInfo(const std::string& filename);
};

} // namespace RiskAnalyst
