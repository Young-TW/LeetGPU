#include <cuda_runtime.h>

#define TILE 16

// C[M x N] = A[M x K] @ W^T (+ optional scaled add into existing C), W stored (N, K).
__global__ void matmul_wt_kernel(const float* A, const float* W, float* C, int M, int K, int N,
                                 float scale, int accumulate) {
    __shared__ float As[TILE][TILE];
    __shared__ float Ws[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int w_col = t * TILE + threadIdx.y;
        int w_row = blockIdx.x * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[(long long)row * K + a_col] : 0.0f;
        Ws[threadIdx.y][threadIdx.x] =
            (w_row < N && w_col < K) ? W[(long long)w_row * K + w_col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += As[threadIdx.y][i] * Ws[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        long long idx = (long long)row * N + col;
        if (accumulate) {
            C[idx] += scale * acc;
        } else {
            C[idx] = scale * acc;
        }
    }
}

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      float* output, int batch, int d_in, int d_out, int rank, float lora_scale) {
    float* t;  // x @ A^T, [batch, rank]
    cudaMalloc(&t, (size_t)batch * rank * sizeof(float));

    dim3 t2(TILE, TILE);
    dim3 g_out((d_out + TILE - 1) / TILE, (batch + TILE - 1) / TILE);
    dim3 g_rank((rank + TILE - 1) / TILE, (batch + TILE - 1) / TILE);

    // output = x @ W^T
    matmul_wt_kernel<<<g_out, t2>>>(x, W, output, batch, d_in, d_out, 1.0f, 0);
    // t = x @ A^T
    matmul_wt_kernel<<<g_rank, t2>>>(x, A, t, batch, d_in, rank, 1.0f, 0);
    // output += lora_scale * t @ B^T
    matmul_wt_kernel<<<g_out, t2>>>(t, B, output, batch, rank, d_out, lora_scale, 1);
    cudaDeviceSynchronize();

    cudaFree(t);
}
