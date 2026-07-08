#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

__global__ void count_kernel(const int* input, int* output, int N, int K) {
    __shared__ int sdata[256];

    int count = 0;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (long long)gridDim.x * blockDim.x) {
        if (input[i] == K) count++;
    }
    sdata[threadIdx.x] = count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && sdata[0] > 0) atomicAdd(output, sdata[0]);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 4096) blocksPerGrid = 4096;

    hipMemset(output, 0, sizeof(int));
    count_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    hipDeviceSynchronize();
}
