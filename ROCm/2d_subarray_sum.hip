#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

__global__ void subarray_2d_sum_kernel(const int* input, int* output, int M, int S_ROW, int E_ROW,
                                       int S_COL, int E_COL) {
    __shared__ int sdata[256];

    int rows = E_ROW - S_ROW + 1;
    int cols = E_COL - S_COL + 1;
    long long count = (long long)rows * cols;

    int sum = 0;
    for (long long t = blockIdx.x * blockDim.x + threadIdx.x; t < count;
         t += (long long)gridDim.x * blockDim.x) {
        int r = S_ROW + (int)(t / cols);
        int c = S_COL + (int)(t % cols);
        sum += input[(long long)r * M + c];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && sdata[0] != 0) atomicAdd(output, sdata[0]);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
    long long count = (long long)(E_ROW - S_ROW + 1) * (E_COL - S_COL + 1);
    int threadsPerBlock = 256;
    long long blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 4096) blocksPerGrid = 4096;

    hipMemset(output, 0, sizeof(int));
    subarray_2d_sum_kernel<<<(int)blocksPerGrid, threadsPerBlock>>>(input, output, M, S_ROW, E_ROW,
                                                                    S_COL, E_COL);
    hipDeviceSynchronize();
}
