cmake_minimum_required(VERSION 3.20)
project(RiskAnalyst VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native")
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
    src/finance/CpuSequentialEngine.cpp
    src/finance/CpuOpenMPEngine.cpp
    src/finance/OptionPricer.cpp
    src/statistics/RiskMetricsCalculator.cpp
    src/benchmark/BenchmarkPipeline.cpp
    src/benchmark/GpuDetector.cpp
)

set(CUDA_SOURCES
    kernels/CudaEngine.cu
    kernels/CudaEngineOptimized.cu
)

add_library(RiskAnalystLib STATIC ${SOURCES} ${CUDA_SOURCES})
target_link_libraries(RiskAnalystLib PUBLIC 
    OpenMP::OpenMP_CXX 
    CUDA::cudart
    CUDA::curand
)

add_executable(RiskAnalyst src/main.cpp)
target_link_libraries(RiskAnalyst PRIVATE RiskAnalystLib)

enable_testing()
add_subdirectory(tests)
