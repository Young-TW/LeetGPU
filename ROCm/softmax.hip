#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>

__global__ void block_max_kernel(const float* input, float* block_max, int N) {
    __shared__ float sdata[256];

    float m = -FLT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        m = fmaxf(m, input[i]);
    }
    sdata[threadIdx.x] = m;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) block_max[blockIdx.x] = sdata[0];
}

__global__ void final_max_kernel(const float* block_max, float* result, int n) {
    __shared__ float sdata[256];

    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, block_max[i]);
    sdata[threadIdx.x] = m;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = sdata[0];
}

__global__ void exp_sum_kernel(const float* input, float* output, const float* max_val, float* sum,
                               int N) {
    __shared__ float sdata[256];

    float m = *max_val;
    float local = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float e = expf(input[i] - m);
        output[i] = e;
        local += e;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(sum, sdata[0]);
}

__global__ void normalize_kernel(float* output, const float* sum, int N) {
    float inv = 1.0f / *sum;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        output[i] *= inv;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    float* scratch;  // [blocksPerGrid partial maxima][max][sum]
    hipMalloc(&scratch, (blocksPerGrid + 2) * sizeof(float));
    float* max_val = scratch + blocksPerGrid;
    float* sum = scratch + blocksPerGrid + 1;
    hipMemset(sum, 0, sizeof(float));

    block_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, scratch, N);
    final_max_kernel<<<1, threadsPerBlock>>>(scratch, max_val, blocksPerGrid);
    exp_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, max_val, sum, N);
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, sum, N);
    hipDeviceSynchronize();
    hipFree(scratch);
}
