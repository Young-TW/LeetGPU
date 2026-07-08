#include "hip/hip_runtime.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define TILE 16

// A: [B, M, K], B: [B, K, N], C: [B, M, N]; FP16 storage, FP32 accumulation.
__global__ void bmm_fp16_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int b = blockIdx.z;
    const half* Ab = A + (long long)b * M * K;
    const half* Bb = B + (long long)b * K * N;
    half* Cb = C + (long long)b * M * N;

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? __half2float(Ab[row * K + a_col]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? __half2float(Bb[b_row * N + col]) : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        Cb[row * N + col] = __float2half(acc);
    }
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, BATCH);

    bmm_fp16_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    hipDeviceSynchronize();
}
