#include <cuda_runtime.h>

// LSD radix sort, 8 bits per pass. Each block owns a tile; per-digit counts are
// scanned digit-major across blocks so a sequential per-block scatter stays stable.

#define TILE 4096
#define BINS 256

__global__ void hist_kernel(const unsigned int* keys, unsigned int* hist, int N, int shift,
                            int numBlocks) {
    __shared__ unsigned int local[BINS];
    for (int i = threadIdx.x; i < BINS; i += blockDim.x) local[i] = 0;
    __syncthreads();

    long long base = (long long)blockIdx.x * TILE;
    for (int i = threadIdx.x; i < TILE; i += blockDim.x) {
        long long idx = base + i;
        if (idx < N) atomicAdd(&local[(keys[idx] >> shift) & 0xFFu], 1u);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BINS; i += blockDim.x) {
        hist[(long long)i * numBlocks + blockIdx.x] = local[i];
    }
}

// Sequential per-tile exclusive scan (thread 0), tile totals to block_sums.
__global__ void scan_tile_kernel(unsigned int* data, unsigned int* block_sums, long long n) {
    if (threadIdx.x != 0) return;
    long long base = (long long)blockIdx.x * TILE;
    unsigned int run = 0;
    for (int i = 0; i < TILE; i++) {
        long long idx = base + i;
        if (idx >= n) break;
        unsigned int v = data[idx];
        data[idx] = run;
        run += v;
    }
    if (block_sums) block_sums[blockIdx.x] = run;
}

__global__ void scan_add_kernel(unsigned int* data, const unsigned int* offsets, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += offsets[i / TILE];
}

static void exclusive_scan(unsigned int* data, long long n) {
    int numBlocks = (int)((n + TILE - 1) / TILE);
    if (numBlocks == 1) {
        scan_tile_kernel<<<1, 32>>>(data, nullptr, n);
        return;
    }
    unsigned int* sums;
    cudaMalloc(&sums, numBlocks * sizeof(unsigned int));
    scan_tile_kernel<<<numBlocks, 32>>>(data, sums, n);
    exclusive_scan(sums, numBlocks);
    int threads = 256;
    scan_add_kernel<<<(int)((n + threads - 1) / threads), threads>>>(data, sums, n);
    cudaFree(sums);
}

__global__ void scatter_kernel(const unsigned int* src, unsigned int* dst,
                               const unsigned int* offsets, int N, int shift, int numBlocks) {
    if (threadIdx.x != 0) return;
    __shared__ unsigned int cursor[BINS];
    for (int d = 0; d < BINS; d++) cursor[d] = offsets[(long long)d * numBlocks + blockIdx.x];

    long long base = (long long)blockIdx.x * TILE;
    for (int i = 0; i < TILE; i++) {
        long long idx = base + i;
        if (idx >= N) break;
        unsigned int key = src[idx];
        unsigned int d = (key >> shift) & 0xFFu;
        dst[cursor[d]++] = key;
    }
}

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {
    int numBlocks = (N + TILE - 1) / TILE;
    long long histSize = (long long)BINS * numBlocks;

    unsigned int *bufA, *bufB, *hist;
    cudaMalloc(&bufA, (size_t)N * sizeof(unsigned int));
    cudaMalloc(&bufB, (size_t)N * sizeof(unsigned int));
    cudaMalloc(&hist, histSize * sizeof(unsigned int));

    const unsigned int* src = input;
    unsigned int* dst = bufA;
    unsigned int* other = bufB;

    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * 8;
        hist_kernel<<<numBlocks, 256>>>(src, hist, N, shift, numBlocks);
        exclusive_scan(hist, histSize);
        scatter_kernel<<<numBlocks, 32>>>(src, dst, hist, N, shift, numBlocks);
        src = dst;
        unsigned int* t = dst == bufA ? bufB : bufA;
        dst = t;
    }

    cudaMemcpy(output, src, (size_t)N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(hist);
}
