#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

// Segmented inclusive scan with the operator
//   (a_sum, a_flag) op (b_sum, b_flag) = (b_flag ? b_sum : a_sum + b_sum, a_flag | b_flag)
// computed per tile (sequentially by thread 0), aggregates combined recursively, then
// the carry from previous tiles is applied to each tile's head (elements before its
// first flag). The exclusive result is derived at the end.

#define TILE 4096

__global__ void tile_scan_kernel(const float* values, const int* flags, float* inclusive,
                                 float* agg_sum, int* agg_flag, long long N) {
    if (threadIdx.x != 0) return;
    long long base = (long long)blockIdx.x * TILE;

    float run = 0.0f;
    int seen_flag = 0;
    for (int i = 0; i < TILE; i++) {
        long long idx = base + i;
        if (idx >= N) break;
        if (flags[idx]) {
            run = values[idx];
            seen_flag = 1;
        } else {
            run += values[idx];
        }
        inclusive[idx] = run;
    }
    agg_sum[blockIdx.x] = run;
    agg_flag[blockIdx.x] = seen_flag;
}

// Exclusive segmented scan of the tile aggregates (sequential; tile count is small
// after one level, and this is called recursively via host code below).
__global__ void agg_scan_kernel(float* agg_sum, int* agg_flag, float* carry_sum, int numTiles) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float run = 0.0f;  // running sum of the open segment before each tile
    for (int i = 0; i < numTiles; i++) {
        carry_sum[i] = run;
        if (agg_flag[i]) {
            run = agg_sum[i];
        } else {
            run += agg_sum[i];
        }
    }
}

__global__ void apply_carry_kernel(const int* flags, float* inclusive, const float* carry_sum,
                                   long long N) {
    if (threadIdx.x != 0) return;
    long long base = (long long)blockIdx.x * TILE;
    float carry = carry_sum[blockIdx.x];
    for (int i = 0; i < TILE; i++) {
        long long idx = base + i;
        if (idx >= N) break;
        if (flags[idx]) return;  // first flag closes the carried segment
        inclusive[idx] += carry;
    }
}

__global__ void to_exclusive_kernel(const int* flags, const float* inclusive, float* output,
                                    long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    output[i] = (flags[i] || i == 0) ? 0.0f : inclusive[i - 1];
}

// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {
    int numTiles = (int)(((long long)N + TILE - 1) / TILE);

    float* inclusive;
    float* agg_sum;
    int* agg_flag;
    float* carry_sum;
    hipMalloc(&inclusive, (size_t)N * sizeof(float));
    hipMalloc(&agg_sum, numTiles * sizeof(float));
    hipMalloc(&agg_flag, numTiles * sizeof(int));
    hipMalloc(&carry_sum, numTiles * sizeof(float));

    tile_scan_kernel<<<numTiles, 32>>>(values, flags, inclusive, agg_sum, agg_flag, N);
    agg_scan_kernel<<<1, 1>>>(agg_sum, agg_flag, carry_sum, numTiles);
    apply_carry_kernel<<<numTiles, 32>>>(flags, inclusive, carry_sum, N);

    int threads = 256;
    long long blocks = ((long long)N + threads - 1) / threads;
    to_exclusive_kernel<<<(int)blocks, threads>>>(flags, inclusive, output, N);
    hipDeviceSynchronize();

    hipFree(inclusive);
    hipFree(agg_sum);
    hipFree(agg_flag);
    hipFree(carry_sum);
}
