#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <curand_kernel.h>

const int MAX_IT = 10000;
const int MAX_THREAD = 10000;


// Rounded operations with different roundings used
__global__ void rounding_operation(float** results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MAX_THREAD) return;


    float valA[4] = { 2, 0, 0, 0 };
    float valB[4] = { 10, 0, 0, 0 };
    float valC[4] = { 4, 0, 0, 0 };
    float valD[4] = { 20, 0, 0, 0 };

    for (int i = 0; i < MAX_IT; i++) {

        float result[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        result[0] = __float2int_rd(valA[0] / valB[0]);
        result[1] = __float2int_ru(valB[0] / valA[0]);
        result[2] = __float2int_rn(valC[0] / valD[0]);
        result[3] = __float2int_rz(valD[0] / valC[0]);

        results[i] = result;
    }
}


// Rounded operations with same roundings used
__global__ void rounding_operation_fixed_rounding(float** results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MAX_THREAD) return;


    float valA[4] = { 2, 0, 0, 0 };
    float valB[4] = { 10, 0, 0, 0 };
    float valC[4] = { 4, 0, 0, 0 };
    float valD[4] = { 20, 0, 0, 0 };

    for (int i = 0; i < MAX_IT; i++) {

        float result[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        result[0] = __float2int_rd(valA[0] / valB[0]);
        result[1] = __float2int_rd(valB[0] / valA[0]);
        result[2] = __float2int_rd(valC[0] / valD[0]);
        result[3] = __float2int_rd(valD[0] / valC[0]);

        results[i] = result;
    }
}

// Execute and compare runtimes
int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    float** d_results;
    cudaMalloc(&d_results, MAX_THREAD * sizeof(float));

    int blockSize = 256;
    int numBlocks = (MAX_THREAD + blockSize - 1) / blockSize;

    auto start = high_resolution_clock::now();

    rounding_operation << <numBlocks, blockSize >> > (d_results);
    cudaDeviceSynchronize();

    auto stop = high_resolution_clock::now();
    auto exec_duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Execution time with random rounding: " << exec_duration.count() << " ms" << std::endl;
    std::cout << std::setprecision(128) << "Average time per calculation: " << (exec_duration.count() * 1000 * 1000) / (MAX_IT * MAX_THREAD) << " ns" << std::endl;
    std::cout << "--------" << std::endl;

    start = high_resolution_clock::now();

    rounding_operation_fixed_rounding << <numBlocks, blockSize >> > (d_results);
    cudaDeviceSynchronize();

    stop = high_resolution_clock::now();
    exec_duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Execution time with fixed rounding: " << exec_duration.count() << " ms" << std::endl;
    std::cout << std::setprecision(128) << "Average time per calculation: " << (exec_duration.count() * 1000 * 1000) / (MAX_IT * MAX_THREAD) << " ns" << std::endl;
    std::cout << "--------" << std::endl;

    cudaFree(d_results);

    return 0;
}
