#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <math.h>

#define TILE 16

// C[M x N] = A[M x K] @ W[K x N]
__global__ void matmul_kernel(const float* A, const float* W, float* C, int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Ws[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int w_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[(long long)row * K + a_col] : 0.0f;
        Ws[threadIdx.y][threadIdx.x] =
            (w_row < K && col < N) ? W[(long long)w_row * N + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += As[threadIdx.y][i] * Ws[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        C[(long long)row * N + col] = acc;
    }
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* hidden, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        hidden[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// x, W_gate, W_up, W_down, output are device pointers
extern "C" void solve(const float* x, const float* W_gate, const float* W_up, const float* W_down,
                      float* output, int M, int d_model, int d_ffn) {
    float *gate, *up, *hidden;
    hipMalloc(&gate, (size_t)M * d_ffn * sizeof(float));
    hipMalloc(&up, (size_t)M * d_ffn * sizeof(float));
    hipMalloc(&hidden, (size_t)M * d_ffn * sizeof(float));

    dim3 t2(TILE, TILE);
    dim3 g_ffn((d_ffn + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 g_out((d_model + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_kernel<<<g_ffn, t2>>>(x, W_gate, gate, M, d_model, d_ffn);
    matmul_kernel<<<g_ffn, t2>>>(x, W_up, up, M, d_model, d_ffn);

    long long n = (long long)M * d_ffn;
    int threads = 256;
    silu_mul_kernel<<<(int)((n + threads - 1) / threads), threads>>>(gate, up, hidden, n);

    matmul_kernel<<<g_out, t2>>>(hidden, W_down, output, M, d_ffn, d_model);
    hipDeviceSynchronize();

    hipFree(gate);
    hipFree(up);
    hipFree(hidden);
}
