#include "hip/hip_runtime.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void dot_fp16_kernel(const half* A, const half* B, float* partial, int N) {
    __shared__ float sdata[256];

    float sum = 0.0f;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (long long)gridDim.x * blockDim.x) {
        sum += __half2float(A[i]) * __half2float(B[i]);
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(partial, sdata[0]);
}

__global__ void to_half_kernel(const float* partial, half* result) {
    *result = __float2half(*partial);
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 4096) blocksPerGrid = 4096;

    float* partial;
    hipMalloc(&partial, sizeof(float));
    hipMemset(partial, 0, sizeof(float));

    dot_fp16_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, partial, N);
    to_half_kernel<<<1, 1>>>(partial, result);
    hipDeviceSynchronize();
    hipFree(partial);
}
