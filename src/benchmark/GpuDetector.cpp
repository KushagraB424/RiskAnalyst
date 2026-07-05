#include "benchmark/GpuDetector.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

namespace RiskAnalyst {

GpuInfo GpuDetector::getGpuInfo() {
    GpuInfo info = {"None", 0, 0, 0, 0, 0, 0, 0, 0};
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        return info;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    info.name = prop.name;
    cudaDriverGetVersion(&info.cudaVersion);
    info.computeCapabilityMajor = prop.major;
    info.computeCapabilityMinor = prop.minor;
    info.clockRateKHz = prop.clockRate;
    info.totalGlobalMemBytes = prop.totalGlobalMem;
    info.multiProcessorCount = prop.multiProcessorCount;
    info.sharedMemPerBlock = prop.sharedMemPerBlock;
    info.warpSize = prop.warpSize;
    
    return info;
}

void GpuDetector::writeHardwareInfo(const std::string& filename) {
    GpuInfo info = getGpuInfo();
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to write hardware info to " << filename << std::endl;
        return;
    }
    
    out << "GPU Name: " << info.name << "\n";
    out << "CUDA Version: " << info.cudaVersion << "\n";
    out << "Compute Capability: " << info.computeCapabilityMajor << "." << info.computeCapabilityMinor << "\n";
    out << "Global Memory (GB): " << static_cast<double>(info.totalGlobalMemBytes) / (1024*1024*1024) << "\n";
    out << "Shared Memory per Block (KB): " << info.sharedMemPerBlock / 1024 << "\n";
    out << "Streaming Multiprocessors: " << info.multiProcessorCount << "\n";
    out << "Warp Size: " << info.warpSize << "\n";
}

} // namespace RiskAnalyst
