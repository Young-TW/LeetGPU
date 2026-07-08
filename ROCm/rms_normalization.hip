#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void sum_squares_kernel(const float* input, float* sum_sq, int N) {
    __shared__ float sdata[256];

    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        sum += input[i] * input[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(sum_sq, sdata[0]);
}

__global__ void normalize_kernel(const float* input, float gamma, float beta, float* output,
                                 const float* sum_sq, int N, float eps) {
    float rms = sqrtf(*sum_sq / N + eps);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        output[i] = gamma * (input[i] / rms) + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    float* sum_sq;
    hipMalloc(&sum_sq, sizeof(float));
    hipMemset(sum_sq, 0, sizeof(float));

    sum_squares_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, sum_sq, N);
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, beta, output, sum_sq, N,
                                                         eps);
    hipDeviceSynchronize();
    hipFree(sum_sq);
}
