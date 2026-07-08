#include <cuda_runtime.h>

// Radix-sort a copy of the input (floats mapped to order-preserving uints),
// then emit the top k in descending order.

#define TILE 4096
#define BINS 256

__device__ inline unsigned int map_key(float f) {
    union {
        float f;
        unsigned int u;
    } v;
    v.f = f;
    unsigned int u = v.u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ inline float unmap_key(unsigned int u) {
    union {
        float f;
        unsigned int u;
    } v;
    v.u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
    return v.f;
}

__global__ void map_kernel(const float* input, unsigned int* keys, int N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) keys[i] = map_key(input[i]);
}

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

__global__ void emit_topk_kernel(const unsigned int* sorted, float* output, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) output[i] = unmap_key(sorted[N - 1 - i]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    int numBlocks = (N + TILE - 1) / TILE;
    long long histSize = (long long)BINS * numBlocks;

    unsigned int *bufA, *bufB, *hist;
    cudaMalloc(&bufA, (size_t)N * sizeof(unsigned int));
    cudaMalloc(&bufB, (size_t)N * sizeof(unsigned int));
    cudaMalloc(&hist, histSize * sizeof(unsigned int));

    int threads = 256;
    map_kernel<<<(int)(((long long)N + threads - 1) / threads), threads>>>(input, bufA, N);

    unsigned int* src = bufA;
    unsigned int* dst = bufB;
    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * 8;
        hist_kernel<<<numBlocks, 256>>>(src, hist, N, shift, numBlocks);
        exclusive_scan(hist, histSize);
        scatter_kernel<<<numBlocks, 32>>>(src, dst, hist, N, shift, numBlocks);
        unsigned int* t = src;
        src = dst;
        dst = t;
    }

    emit_topk_kernel<<<(k + threads - 1) / threads, threads>>>(src, output, N, k);
    cudaDeviceSynchronize();
    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(hist);
}
