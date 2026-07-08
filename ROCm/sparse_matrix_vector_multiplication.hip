#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

// A is stored dense in row-major order; one block per row with a shared reduction.
__global__ void spmv_kernel(const float* A, const float* x, float* y, int M, int N) {
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float a = A[(long long)row * N + j];
        if (a != 0.0f) sum += a * x[j];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = sdata[0];
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    spmv_kernel<<<M, 256>>>(A, x, y, M, N);
    hipDeviceSynchronize();
}
