#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <math.h>

// output[i] = SiLU(x1[i]) * x2[i], where x1 is the first half and x2 the second.
__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < halfN) {
        float x1 = input[i];
        float x2 = input[i + halfN];
        float silu = x1 / (1.0f + expf(-x1));
        output[i] = silu * x2;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    hipDeviceSynchronize();
}
