#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

#define TILE 16

// A: [BATCH, M, K], B: [BATCH, K, N], C: [BATCH, M, N] (row-major)
__global__ void bmm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int b = blockIdx.z;
    const float* Ab = A + (long long)b * M * K;
    const float* Bb = B + (long long)b * K * N;
    float* Cb = C + (long long)b * M * N;

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? Ab[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? Bb[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        Cb[row * N + col] = acc;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, BATCH);

    bmm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    hipDeviceSynchronize();
}
