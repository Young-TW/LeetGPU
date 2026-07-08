#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int base = idx * 4;
        image[base + 0] = 255 - image[base + 0];
        image[base + 1] = 255 - image[base + 1];
        image[base + 2] = 255 - image[base + 2];
        // alpha stays unchanged
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    hipDeviceSynchronize();
}
