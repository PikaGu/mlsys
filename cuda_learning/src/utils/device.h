#ifndef _DEVICE_H
#define _DEVICE_H

#include "utils/common.h"

#include <cuda_runtime.h>
#include <iostream>

void deviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (auto i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "Device name: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "----------------------------------------------------------------\n";
    }
}

#endif
