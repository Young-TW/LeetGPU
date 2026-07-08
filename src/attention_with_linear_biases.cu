#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define THREADS 128
#define MAX_D 1024

// score(i, j) = Q_i . K_j / sqrt(d) + alpha * (i - j); softmax row-wise over all keys.
__global__ void alibi_attention_kernel(const float* Q, const float* K, const float* V,
                                       float* output, int M, int N, int d, float alpha) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* q = smem;          // d
    float* acc = smem + d;    // d
    __shared__ float sdata[THREADS];

    for (int i = tid; i < d; i += blockDim.x) {
        q[i] = Q[(long long)row * d + i];
        acc[i] = 0.0f;
    }
    __syncthreads();

    float scale = (1.0f / sqrtf((float)d));

    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += q[i] * K[(long long)j * d + i];
        s = s * scale + alpha * (row - j);
        local_max = fmaxf(local_max, s);
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
    for (int j = tid; j < N; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += q[i] * K[(long long)j * d + i];
        s = s * scale + alpha * (row - j);
        float e = expf(s - row_max);
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
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) {
    size_t smem = 2 * (size_t)d * sizeof(float);
    alibi_attention_kernel<<<M, THREADS, smem>>>(Q, K, V, output, M, N, d, alpha);
    cudaDeviceSynchronize();
}
