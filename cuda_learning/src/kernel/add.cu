#include "utils/common.h"
#include <iostream>
#include <stdio.h>

__global__
void helloWorld() {
    printf("Hello world: %d %d %d\n", threadIdx.x, blockIdx.x, blockDim.x);
}

__global__
void cuda_sum_1d(float* A, float* B, float* C, size_t size) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__
void cuda_sum_2d(float* A, float*B, float* C, size_t nx, size_t ny) {

}

void sum1d() {
    auto nElem = 4096 * 4096;
    auto nBytes = nElem * sizeof(float);

    auto A = (float*) malloc(nBytes);
    auto B = (float*) malloc(nBytes);
    auto C = (float*) malloc(nBytes);
    auto C_GPU = (float*) malloc(nBytes);
    initialData(A, nElem);
    initialData(B, nElem);

    auto cpu_start = cpuSecond();
    for (auto i = 0; i < nElem; i++) {
        C[i] = A[i] + B[i];
    }
    std::cout << "CPU cost: " << cpuSecond() - cpu_start << "\n";
    
    float* dA;
    float* dB;
    float* dC;
    CHECK(cudaMalloc((float**) &dA, nBytes));
    CHECK(cudaMalloc((float**) &dB, nBytes));
    CHECK(cudaMalloc((float**) &dC, nBytes));
    CHECK(cudaMemcpy(dA, A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, nBytes, cudaMemcpyHostToDevice));

    dim3 block(1024);
    std::cout << "block: " << block.x << "," << block.y << "\n";
    dim3 grid = (nElem + block.x - 1) / block.x;
    std::cout << "grid: " << grid.x << "," << grid.y << "\n";

    auto gpu_start = cpuSecond();
    cuda_sum_1d<<<grid, block>>>(dA, dB, dC, nElem);
    CHECK(cudaDeviceSynchronize());
    std::cout << "GPU cost: " << cpuSecond() - gpu_start << "\n";

    CHECK(cudaMemcpy(C_GPU, dC, nBytes, cudaMemcpyDeviceToHost));

    for (auto i = 0; i < nElem; i++) {
        if (C[i] != C_GPU[i]) {
            std::cout << "diff! " << i << "\n";
            break;
        }
    }

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    free(A);
    free(B);
    free(C);
    free(C_GPU);
}

void sum2d() {

}