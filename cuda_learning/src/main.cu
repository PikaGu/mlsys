#include "kernel/add.cu"
#include "utils/common.h"
#include "utils/device.h"

#include <cuda_runtime.h>
#include <iostream>
using namespace std;

int main() {
    // sum1d();
    // deviceInfo();
    sum2d();
    cudaDeviceReset();
    return 0;
}