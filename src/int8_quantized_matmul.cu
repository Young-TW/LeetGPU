#include <cuda_runtime.h>
#include <stdint.h>

__global__ void quant_matmul_kernel(const int8_t* A, const int8_t* B, int8_t* C, int M, int N,
                                    int K, float scale_A, float scale_B, float scale_C,
                                    int zero_point_A, int zero_point_B, int zero_point_C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    int acc = 0;
    for (int k = 0; k < K; k++) {
        acc += ((int)A[(long long)row * K + k] - zero_point_A) *
               ((int)B[(long long)k * N + col] - zero_point_B);
    }
    float scaled = (float)acc * scale_A * scale_B / scale_C;
    int q = (int)rintf(scaled) + zero_point_C;
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    C[(long long)row * N + col] = (int8_t)q;
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    quant_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, scale_A, scale_B,
                                                            scale_C, zero_point_A, zero_point_B,
                                                            zero_point_C);
    cudaDeviceSynchronize();
}
