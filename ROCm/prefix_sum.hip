#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

#define BLOCK 512  // threads per block; each block scans 2*BLOCK elements

// Inclusive scan of one tile (2*BLOCK elements) per block; the tile total goes to block_sums.
__global__ void scan_block_kernel(const float* input, float* output, float* block_sums, int N) {
    __shared__ float temp[2 * BLOCK];
    int tid = threadIdx.x;
    long long base = (long long)blockIdx.x * 2 * BLOCK;

    temp[2 * tid] = (base + 2 * tid < N) ? input[base + 2 * tid] : 0.0f;
    temp[2 * tid + 1] = (base + 2 * tid + 1 < N) ? input[base + 2 * tid + 1] : 0.0f;
    __syncthreads();

    // up-sweep
    int offset = 1;
    for (int d = BLOCK; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
        __syncthreads();
    }

    float total = temp[2 * BLOCK - 1];
    if (tid == 0) {
        if (block_sums) block_sums[blockIdx.x] = total;
        temp[2 * BLOCK - 1] = 0.0f;
    }
    __syncthreads();

    // down-sweep (exclusive)
    for (int d = 1; d <= BLOCK; d <<= 1) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // convert exclusive to inclusive by adding the input element
    if (base + 2 * tid < N)
        output[base + 2 * tid] = temp[2 * tid] + input[base + 2 * tid];
    if (base + 2 * tid + 1 < N)
        output[base + 2 * tid + 1] = temp[2 * tid + 1] + input[base + 2 * tid + 1];
}

// Add the (inclusive) scanned sum of all previous blocks to each element.
__global__ void add_offsets_kernel(float* output, const float* scanned_sums, int N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long block = i / (2 * BLOCK);
    if (i < N && block > 0) {
        output[i] += scanned_sums[block - 1];
    }
}

static void scan_recursive(const float* input, float* output, int N) {
    int tile = 2 * BLOCK;
    int numBlocks = (N + tile - 1) / tile;

    if (numBlocks == 1) {
        scan_block_kernel<<<1, BLOCK>>>(input, output, nullptr, N);
        return;
    }

    float* block_sums;
    float* scanned_sums;
    hipMalloc(&block_sums, numBlocks * sizeof(float));
    hipMalloc(&scanned_sums, numBlocks * sizeof(float));

    scan_block_kernel<<<numBlocks, BLOCK>>>(input, output, block_sums, N);
    scan_recursive(block_sums, scanned_sums, numBlocks);

    int threads = 256;
    long long total = N;
    int addBlocks = (int)((total + threads - 1) / threads);
    add_offsets_kernel<<<addBlocks, threads>>>(output, scanned_sums, N);

    hipFree(block_sums);
    hipFree(scanned_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scan_recursive(input, output, N);
    hipDeviceSynchronize();
}
