#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// W4A16: y = x @ W^T, W dequantized on the fly from packed INT4 nibbles.
// Byte w_q[n][i] holds w[n, 2i] in the high nibble and w[n, 2i+1] in the low
// nibble; signed value = nibble - 8. Group-wise scales along K.
__global__ void w4a16_kernel(const __half* x, const uint8_t* w_q, const __half* scales, __half* y,
                             int M, int N, int K, int group_size) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int groups = K / group_size;
    const uint8_t* wrow = w_q + (long long)n * (K / 2);
    const __half* srow = scales + (long long)n * groups;
    const __half* xrow = x + (long long)m * K;

    float acc = 0.0f;
    for (int g = 0; g < groups; g++) {
        float scale = __half2float(srow[g]);
        int k0 = g * group_size;
        float part = 0.0f;
        for (int k = k0; k < k0 + group_size; k += 2) {
            uint8_t byte = wrow[k / 2];
            int hi = (byte >> 4) - 8;   // w[n, k]
            int lo = (byte & 0xF) - 8;  // w[n, k+1]
            part += __half2float(xrow[k]) * hi;
            part += __half2float(xrow[k + 1]) * lo;
        }
        acc += part * scale;
    }
    y[(long long)m * N + n] = __float2half(acc);
}

// x, w_q, scales, y are device pointers
extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    w4a16_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, w_q, scales, y, M, N, K, group_size);
    cudaDeviceSynchronize();
}
