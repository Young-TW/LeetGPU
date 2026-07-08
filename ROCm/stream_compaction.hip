#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

// Exclusive scan of the predicate (A[i] > 0) gives each kept element its output
// slot. Scan is tile-sequential per block with recursive aggregate scan.

#define TILE 4096

__global__ void tile_scan_kernel(const float* A, unsigned int* pos, unsigned int* tile_sums,
                                 long long N) {
    if (threadIdx.x != 0) return;
    long long base = (long long)blockIdx.x * TILE;
    unsigned int run = 0;
    for (int i = 0; i < TILE; i++) {
        long long idx = base + i;
        if (idx >= N) break;
        pos[idx] = run;
        if (A[idx] > 0.0f) run++;
    }
    if (tile_sums) tile_sums[blockIdx.x] = run;
}

__global__ void agg_scan_kernel(unsigned int* sums, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int run = 0;
    for (int i = 0; i < n; i++) {
        unsigned int v = sums[i];
        sums[i] = run;
        run += v;
    }
}

__global__ void scatter_kernel(const float* A, const unsigned int* pos,
                               const unsigned int* tile_offsets, float* out, long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (A[i] > 0.0f) {
        out[tile_offsets[i / TILE] + pos[i]] = A[i];
    }
}

// A, out are device pointers
extern "C" void solve(const float* A, int N, float* out) {
    int numTiles = (int)(((long long)N + TILE - 1) / TILE);

    unsigned int* pos;
    unsigned int* tile_sums;
    hipMalloc(&pos, (size_t)N * sizeof(unsigned int));
    hipMalloc(&tile_sums, numTiles * sizeof(unsigned int));

    hipMemset(out, 0, (size_t)N * sizeof(float));  // out is stated pre-zeroed, but be safe

    tile_scan_kernel<<<numTiles, 32>>>(A, pos, tile_sums, N);
    agg_scan_kernel<<<1, 1>>>(tile_sums, numTiles);

    int threads = 256;
    long long blocks = ((long long)N + threads - 1) / threads;
    scatter_kernel<<<(int)blocks, threads>>>(A, pos, tile_sums, out, N);
    hipDeviceSynchronize();

    hipFree(pos);
    hipFree(tile_sums);
}
