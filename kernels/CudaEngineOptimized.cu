#include "cuda/CudaEngineOptimized.cuh"
#include <curand_kernel.h>
#include <chrono>
#include <iostream>

namespace RiskAnalyst {

#define CUDA_CHECK(call)     do {         cudaError_t err = call;         if (err != cudaSuccess) {             std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl;             exit(EXIT_FAILURE);         }     } while (0)

__constant__ double d_params[5];

__global__ void init_curand_opt_kernel(unsigned int seed, curandState_t* states, size_t numPaths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPaths) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void simulate_and_price_fused_kernel(
    curandState_t* states, size_t numSteps, size_t numPaths, 
    OptionType type, double discountFactor, double* d_payoffs) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPaths) {
        double S0 = d_params[0];
        double r = d_params[1];
        double v = d_params[2];
        double T = d_params[3];
        double K = d_params[4];

        double dt = T / (double)numSteps;
        double drift = (r - 0.5 * v * v) * dt;
        double vol = v * sqrt(dt);

        curandState_t localState = states[idx];
        double currentPrice = S0;
        double sumPrices = 0.0;
        
        for (size_t step = 0; step < numSteps; ++step) {
            double Z = curand_normal_double(&localState);
            currentPrice *= exp(drift + vol * Z);
            sumPrices += currentPrice;
        }
        
        double payoff = 0.0;
        if (type == OptionType::EuropeanCall) {
            payoff = fmax(currentPrice - K, 0.0);
        } else if (type == OptionType::EuropeanPut) {
            payoff = fmax(K - currentPrice, 0.0);
        } else if (type == OptionType::AsianCall) {
            payoff = fmax((sumPrices / (double)numSteps) - K, 0.0);
        }

        d_payoffs[idx] = payoff * discountFactor;
    }
}

__global__ void reduce_sum_kernel(const double* d_in, double* d_out, size_t n) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    double mySum = (i < n) ? d_in[i] : 0.0;
    if (i + blockDim.x < n) mySum += d_in[i + blockDim.x];
    
    sdata[tid] = mySum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

SimulationResult CudaEngineOptimized::runSimulation(const Config& config) {
    cudaEvent_t totalStart, totalStop, h2dStart, h2dStop, kernelStart, kernelStop, d2hStart, d2hStop;
    cudaEventCreate(&totalStart); cudaEventCreate(&totalStop);
    cudaEventCreate(&h2dStart); cudaEventCreate(&h2dStop);
    cudaEventCreate(&kernelStart); cudaEventCreate(&kernelStop);
    cudaEventCreate(&d2hStart); cudaEventCreate(&d2hStop);
    cudaEventRecord(totalStart);

    const size_t numPaths = config.simulation.numPaths;
    const size_t numSteps = config.simulation.numSteps;
    const double discountFactor = exp(-config.market.riskFreeRate * config.market.timeToMaturity);

    double h_params[5] = {config.market.spotPrice, config.market.riskFreeRate, config.market.volatility, config.market.timeToMaturity, config.market.strikePrice};

    cudaEventRecord(h2dStart);
    CUDA_CHECK(cudaMemcpyToSymbol(d_params, h_params, 5 * sizeof(double)));
    cudaEventRecord(h2dStop);

    int blockSize = 0, minGridSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, simulate_and_price_fused_kernel, 0, 0));
    if (blockSize == 0) blockSize = 256;
    int gridSize = (numPaths + blockSize - 1) / blockSize;

    curandState_t* d_states = nullptr; double* d_payoffs = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_states, numPaths * sizeof(curandState_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_payoffs, numPaths * sizeof(double)));

    double* h_payoffs_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&h_payoffs_pinned, numPaths * sizeof(double), cudaHostAllocDefault));

    cudaEventRecord(kernelStart);
    init_curand_opt_kernel<<<gridSize, blockSize>>>(config.simulation.seed, d_states, numPaths);
    simulate_and_price_fused_kernel<<<gridSize, blockSize>>>(d_states, numSteps, numPaths, config.simulation.optionType, discountFactor, d_payoffs);
    
    int reduceBlockSize = 256;
    int reduceGridSize = (numPaths + (reduceBlockSize * 2) - 1) / (reduceBlockSize * 2);
    double* d_partialSums = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_partialSums, reduceGridSize * sizeof(double)));
    
    size_t sharedMemSize = reduceBlockSize * sizeof(double);
    reduce_sum_kernel<<<reduceGridSize, reduceBlockSize, sharedMemSize>>>(d_payoffs, d_partialSums, numPaths);
    
    double* h_partialSums = new double[reduceGridSize];
    CUDA_CHECK(cudaMemcpy(h_partialSums, d_partialSums, reduceGridSize * sizeof(double), cudaMemcpyDeviceToHost));
    double gpuTotalSum = 0.0;
    for (int i = 0; i < reduceGridSize; ++i) gpuTotalSum += h_partialSums[i];
    cudaEventRecord(kernelStop);

    cudaEventRecord(d2hStart);
    CUDA_CHECK(cudaMemcpy(h_payoffs_pinned, d_payoffs, numPaths * sizeof(double), cudaMemcpyDeviceToHost));
    cudaEventRecord(d2hStop);

    SimulationResult result;
    result.payoffs.assign(h_payoffs_pinned, h_payoffs_pinned + numPaths);
    result.gpuSum = gpuTotalSum;

    CUDA_CHECK(cudaFreeHost(h_payoffs_pinned));
    CUDA_CHECK(cudaFree(d_states)); CUDA_CHECK(cudaFree(d_payoffs)); CUDA_CHECK(cudaFree(d_partialSums));
    delete[] h_partialSums;

    cudaEventRecord(totalStop); cudaEventSynchronize(totalStop);

    float h2dMs=0, kernelMs=0, d2hMs=0, totalMs=0;
    cudaEventElapsedTime(&h2dMs, h2dStart, h2dStop);
    cudaEventElapsedTime(&kernelMs, kernelStart, kernelStop);
    cudaEventElapsedTime(&d2hMs, d2hStart, d2hStop);
    cudaEventElapsedTime(&totalMs, totalStart, totalStop);

    result.totalSimulations = numPaths; result.timeSteps = numSteps; result.optionType = config.simulation.optionType;
    result.hostToDeviceTimeMs = h2dMs; result.kernelTimeMs = kernelMs; result.deviceToHostTimeMs = d2hMs; result.totalExecutionTimeMs = totalMs;
    result.throughput = (numPaths / (totalMs / 1000.0));
    result.blockSize = blockSize;
    result.gridSize = gridSize;
    
    return result;
}

} // namespace RiskAnalyst
