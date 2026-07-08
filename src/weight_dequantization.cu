#include <cuda_runtime.h>

// Y[i][j] = X[i][j] * S[i / T][j / T]
__global__ void dequant_kernel(const float* X, const float* S, float* Y, int M, int N,
                               int TILE_SIZE, int s_cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;

    float scale = S[(i / TILE_SIZE) * s_cols + (j / TILE_SIZE)];
    Y[(long long)i * N + j] = X[(long long)i * N + j] * scale;
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    int s_cols = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    dequant_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, S, Y, M, N, TILE_SIZE, s_cols);
    cudaDeviceSynchronize();
}
