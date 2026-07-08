#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define THREADS 128
#define MAX_D 128

// Row i attends only to keys j in [i - window_size, i + window_size] within bounds.
__global__ void sliding_window_attention_kernel(const float* Q, const float* K, const float* V,
                                                float* output, int M, int d, int window_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lo = row - window_size < 0 ? 0 : row - window_size;
    int hi = row + window_size > M - 1 ? M - 1 : row + window_size;

    __shared__ float q[MAX_D];
    __shared__ float acc[MAX_D];
    __shared__ float sdata[THREADS];

    for (int i = tid; i < d; i += blockDim.x) {
        q[i] = Q[(long long)row * d + i];
        acc[i] = 0.0f;
    }
    __syncthreads();

    float scale = (1.0f / sqrtf((float)d));

    float local_max = -FLT_MAX;
    for (int j = lo + tid; j <= hi; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += q[i] * K[(long long)j * d + i];
        local_max = fmaxf(local_max, s * scale);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = lo + tid; j <= hi; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += q[i] * K[(long long)j * d + i];
        float e = expf(s * scale - row_max);
        local_sum += e;
        for (int i = 0; i < d; i++) atomicAdd(&acc[i], e * V[(long long)j * d + i]);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int i = tid; i < d; i += blockDim.x) {
        output[(long long)row * d + i] = acc[i] / row_sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int window_size) {
    sliding_window_attention_kernel<<<M, THREADS>>>(Q, K, V, output, M, d, window_size);
    cudaDeviceSynchronize();
}
