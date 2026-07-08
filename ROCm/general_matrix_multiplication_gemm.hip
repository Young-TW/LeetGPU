#include "hip/hip_runtime.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define TILE 16

// C = alpha * (A x B) + beta * C, FP16 in/out, FP32 accumulation.
// A: M x K, B: K x N, C: M x N (row-major)
__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K,
                            float alpha, float beta) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? __half2float(A[row * K + a_col]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? __half2float(B[b_row * N + col]) : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float c0 = __half2float(C[row * N + col]);
        C[row * N + col] = __float2half(alpha * acc + beta * c0);
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    hipDeviceSynchronize();
}
