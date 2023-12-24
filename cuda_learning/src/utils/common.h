#ifndef _COMMON_H
#define _COMMON_H

#include <stdio.h>

static void handleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK(err) (handleError(err, __FILE__, __LINE__))

void initialData(float* ip, int size) {
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xffff) / 1000.0f;
    }
}

#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6;
}

#endif