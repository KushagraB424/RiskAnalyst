#include "cuda/CudaEngine.cuh"
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <chrono>
#include <iostream>

namespace RiskAnalyst {

// Error checking macro
#define CUDA_CHECK(call)     do {         cudaError_t err = call;         if (err != cudaSuccess) {             std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " "" << cudaGetErrorString(err) << """ << std::endl;             exit(EXIT_FAILURE);         }     } while (0)

__global__ void init_curand_kernel(unsigned int seed, curandState_t* states, size_t numPaths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPaths) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void simulate_paths_kernel(
    curandState_t* states, 
    double S0, double drift, double vol, 
    size_t numSteps, size_t numPaths, 
    double* d_finalPrices, double* d_averagePrices) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPaths) {
        curandState_t localState = states[idx];
        double currentPrice = S0;
        double sumPrices = 0.0;
        
        for (size_t step = 0; step < numSteps; ++step) {
            double Z = curand_normal_double(&localState);
            currentPrice *= exp(drift + vol * Z);
            sumPrices += currentPrice;
        }
        
        states[idx] = localState; // Save state (though not reused here)
        d_finalPrices[idx] = currentPrice;
        d_averagePrices[idx] = sumPrices / (double)numSteps;
    }
}

__global__ void calculate_payoffs_kernel(
    const double* d_finalPrices, 
    const double* d_averagePrices, 
    double strikePrice, 
    OptionType type, 
    double discountFactor, 
    size_t numPaths, 
    double* d_payoffs) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPaths) {
        double payoff = 0.0;
        if (type == OptionType::EuropeanCall) {
            payoff = fmax(d_finalPrices[idx] - strikePrice, 0.0);
        } else if (type == OptionType::EuropeanPut) {
            payoff = fmax(strikePrice - d_finalPrices[idx], 0.0);
        } else if (type == OptionType::AsianCall) {
            payoff = fmax(d_averagePrices[idx] - strikePrice, 0.0);
        }
        d_payoffs[idx] = payoff * discountFactor;
    }
}

SimulationResult CudaEngine::runSimulation(const Config& config) {
    auto start = std::chrono::high_resolution_clock::now();

    const size_t numPaths = config.simulation.numPaths;
    const size_t numSteps = config.simulation.numSteps;
    const double S0 = config.market.spotPrice;
    const double K = config.market.strikePrice;
    const double r = config.market.riskFreeRate;
    const double v = config.market.volatility;
    const double T = config.market.timeToMaturity;
    
    const double dt = T / (double)numSteps;
    const double drift = (r - 0.5 * v * v) * dt;
    const double vol = v * sqrt(dt);
    const double discountFactor = exp(-r * T);

    // Block & Grid config
    int blockSize = 256;
    int gridSize = (numPaths + blockSize - 1) / blockSize;

    // Device allocations (Baseline approach without specific optimizations yet)
    curandState_t* d_states = nullptr;
    double *d_finalPrices = nullptr, *d_averagePrices = nullptr, *d_payoffs = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_states, numPaths * sizeof(curandState_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_finalPrices, numPaths * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_averagePrices, numPaths * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_payoffs, numPaths * sizeof(double)));

    // Initialize RNG
    init_curand_kernel<<<gridSize, blockSize>>>(config.simulation.seed, d_states, numPaths);
    CUDA_CHECK(cudaGetLastError());

    // Generate Paths
    simulate_paths_kernel<<<gridSize, blockSize>>>(d_states, S0, drift, vol, numSteps, numPaths, d_finalPrices, d_averagePrices);
    CUDA_CHECK(cudaGetLastError());

    // Compute Payoffs
    calculate_payoffs_kernel<<<gridSize, blockSize>>>(d_finalPrices, d_averagePrices, K, config.simulation.optionType, discountFactor, numPaths, d_payoffs);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    SimulationResult result;
    result.finalPrices.resize(numPaths);
    result.payoffs.resize(numPaths);
    
    CUDA_CHECK(cudaMemcpy(result.finalPrices.data(), d_finalPrices, numPaths * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.payoffs.data(), d_payoffs, numPaths * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_finalPrices));
    CUDA_CHECK(cudaFree(d_averagePrices));
    CUDA_CHECK(cudaFree(d_payoffs));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    result.totalSimulations = numPaths;
    result.timeSteps = numSteps;
    result.optionType = config.simulation.optionType;
    result.totalExecutionTimeMs = elapsed.count();
    result.throughput = (numPaths / (result.totalExecutionTimeMs / 1000.0));

    return result;
}

} // namespace RiskAnalyst
