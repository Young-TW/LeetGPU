#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

#define TILE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = acc;
}

__global__ void identity_kernel(float* A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) A[idx] = (idx / N == idx % N) ? 1.0f : 0.0f;
}

// output = input^P via binary exponentiation
extern "C" void solve(const float* input, float* output, int N, int P) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *base, *tmp, *result;
    hipMalloc(&base, bytes);
    hipMalloc(&tmp, bytes);
    hipMalloc(&result, bytes);

    hipMemcpy(base, input, bytes, hipMemcpyDeviceToDevice);
    identity_kernel<<<(N * N + 255) / 256, 256>>>(result, N);

    dim3 t2(TILE, TILE);
    dim3 g2((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    int p = P;
    while (p > 0) {
        if (p & 1) {
            matmul_kernel<<<g2, t2>>>(result, base, tmp, N);
            float* t = result;
            result = tmp;
            tmp = t;
        }
        p >>= 1;
        if (p > 0) {
            matmul_kernel<<<g2, t2>>>(base, base, tmp, N);
            float* t = base;
            base = tmp;
            tmp = t;
        }
    }

    hipMemcpy(output, result, bytes, hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();
    hipFree(base);
    hipFree(tmp);
    hipFree(result);
}
