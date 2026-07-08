#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <math.h>

// One block per feature channel; reduce mean/var over the batch dimension.
__global__ void batch_norm_kernel(const float* input, const float* gamma, const float* beta,
                                  float* output, int N, int C, float eps) {
    __shared__ float sdata[256];
    int c = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) sum += input[(long long)i * C + c];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / N;
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float d = input[(long long)i * C + c] - mean;
        var_sum += d * d;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = (1.0f / sqrtf(sdata[0] / N + eps));
    __syncthreads();

    float g = gamma[c], b = beta[c];
    for (int i = tid; i < N; i += blockDim.x) {
        long long idx = (long long)i * C + c;
        output[idx] = g * (input[idx] - mean) * inv_std + b;
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {
    batch_norm_kernel<<<C, 256>>>(input, gamma, beta, output, N, C, eps);
    hipDeviceSynchronize();
}
