#include <cuda_runtime.h>
#include <math.h>

#define THREADS 128
#define MAX_D 256

// output[n] = sum_{m<=n} gamma^(n-m) * (Q[n].K[m]/sqrt(d)) * V[m] — no softmax.
__global__ void retention_kernel(const float* Q, const float* K, const float* V, float* output,
                                 int seq_len, int d_model, float gamma) {
    extern __shared__ float smem[];
    float* q = smem;                // d_model
    float* acc = smem + d_model;    // d_model

    int n = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = tid; i < d_model; i += blockDim.x) {
        q[i] = Q[(long long)n * d_model + i];
        acc[i] = 0.0f;
    }
    __syncthreads();

    float scale = (1.0f / sqrtf((float)d_model));

    for (int m = tid; m <= n; m += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < d_model; i++) s += q[i] * K[(long long)m * d_model + i];
        float w = powf(gamma, (float)(n - m)) * s * scale;
        for (int i = 0; i < d_model; i++) atomicAdd(&acc[i], w * V[(long long)m * d_model + i]);
    }
    __syncthreads();

    for (int i = tid; i < d_model; i += blockDim.x) {
        output[(long long)n * d_model + i] = acc[i];
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma) {
    size_t smem = 2 * (size_t)d_model * sizeof(float);
    retention_kernel<<<seq_len, THREADS, smem>>>(Q, K, V, output, seq_len, d_model, gamma);
    cudaDeviceSynchronize();
}
