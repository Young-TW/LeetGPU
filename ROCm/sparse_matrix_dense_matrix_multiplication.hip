#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

#define TILE 16

// A (sparse, stored dense) M x N times B (dense) N x K -> C M x K.
__global__ void spmm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < N) ? A[(long long)row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < N && col < K) ? B[(long long)b_row * K + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < K) {
        C[(long long)row * K + col] = acc;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    spmm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    hipDeviceSynchronize();
}
