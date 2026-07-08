#include <cuda_runtime.h>
#include <math.h>

// One block per (batch, group): reduce mean/var over (C/G) * H * W elements,
// then normalize with per-channel gamma/beta.
__global__ void group_norm_kernel(const float* X, const float* gamma, const float* beta, float* Y,
                                  int C, int H, int W, int G, float eps) {
    __shared__ float sdata[256];
    int n = blockIdx.x;
    int g = blockIdx.y;
    int tid = threadIdx.x;

    int ch_per_group = C / G;
    long long hw = (long long)H * W;
    long long group_size = ch_per_group * hw;
    const float* base = X + ((long long)n * C + (long long)g * ch_per_group) * hw;

    float sum = 0.0f;
    for (long long i = tid; i < group_size; i += blockDim.x) sum += base[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / group_size;
    __syncthreads();

    float var_sum = 0.0f;
    for (long long i = tid; i < group_size; i += blockDim.x) {
        float d = base[i] - mean;
        var_sum += d * d;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = (1.0f / sqrtf(sdata[0] / group_size + eps));
    __syncthreads();

    float* out = Y + ((long long)n * C + (long long)g * ch_per_group) * hw;
    for (long long i = tid; i < group_size; i += blockDim.x) {
        int c = g * ch_per_group + (int)(i / hw);
        out[i] = gamma[c] * (base[i] - mean) * inv_std + beta[c];
    }
}

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N,
                      int C, int H, int W, int G, float eps) {
    dim3 grid(N, G);
    group_norm_kernel<<<grid, 256>>>(X, gamma, beta, Y, C, H, W, G, eps);
    cudaDeviceSynchronize();
}
