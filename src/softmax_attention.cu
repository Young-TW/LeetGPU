#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define THREADS 128
#define MAX_D 128

// One block per query row. Two passes over the keys: max score, then exp-sum
// while accumulating the weighted values in shared memory.
__global__ void attention_row_kernel(const float* Q, const float* K, const float* V, float* output,
                                     int M, int N, int d) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float q[MAX_D];
    __shared__ float acc[MAX_D];
    __shared__ float sdata[THREADS];

    for (int i = tid; i < d; i += blockDim.x) {
        q[i] = Q[row * d + i];
        acc[i] = 0.0f;
    }
    __syncthreads();

    float scale = (1.0f / sqrtf((float)d));

    // Pass 1: row max of the scores
    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += q[i] * K[j * d + i];
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

    // Pass 2: exp-sum and weighted value accumulation
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < d; i++) s += q[i] * K[j * d + i];
        float e = expf(s * scale - row_max);
        local_sum += e;
        for (int i = 0; i < d; i++) atomicAdd(&acc[i], e * V[j * d + i]);
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
        output[row * d + i] = acc[i] / row_sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    attention_row_kernel<<<M, THREADS>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();
}
