#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <limits.h>

// Fixed-window max sum: prefix sums, then max over prefix[i+w] - prefix[i].

__global__ void prefix_kernel(const int* input, long long* prefix, int N) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    prefix[0] = 0;
    for (int i = 0; i < N; i++) prefix[i + 1] = prefix[i] + input[i];
}

__global__ void window_max_kernel(const long long* prefix, int* output, int N, int w) {
    __shared__ int sdata[256];

    int best = INT_MIN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i + w <= N;
         i += gridDim.x * blockDim.x) {
        long long s = prefix[i + w] - prefix[i];
        if (s > best) best = (int)s;
    }
    sdata[threadIdx.x] = best;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sdata[threadIdx.x + s] > sdata[threadIdx.x])
            sdata[threadIdx.x] = sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(output, sdata[0]);
}

__global__ void init_min_kernel(int* output) {
    *output = INT_MIN;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    long long* prefix;
    hipMalloc(&prefix, (N + 1) * sizeof(long long));

    prefix_kernel<<<1, 1>>>(input, prefix, N);
    init_min_kernel<<<1, 1>>>(output);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    window_max_kernel<<<blocks, threads>>>(prefix, output, N, window_size);
    hipDeviceSynchronize();
    hipFree(prefix);
}
