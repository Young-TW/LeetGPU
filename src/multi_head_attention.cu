#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define THREADS 128
#define MAX_DK 1024

// One block per (query row, head). Two passes over the keys with the head's
// slice of columns [head*d_k, (head+1)*d_k).
__global__ void mha_row_kernel(const float* Q, const float* K, const float* V, float* output,
                               int N, int d_model, int h) {
    int row = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int d_k = d_model / h;
    int off = head * d_k;

    extern __shared__ float smem[];
    float* q = smem;            // d_k
    float* acc = smem + d_k;    // d_k
    __shared__ float sdata[THREADS];

    for (int i = tid; i < d_k; i += blockDim.x) {
        q[i] = Q[row * d_model + off + i];
        acc[i] = 0.0f;
    }
    __syncthreads();

    float scale = (1.0f / sqrtf((float)d_k));

    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < d_k; i++) s += q[i] * K[j * d_model + off + i];
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
    for (int j = tid; j < N; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < d_k; i++) s += q[i] * K[j * d_model + off + i];
        float e = expf(s * scale - row_max);
        local_sum += e;
        for (int i = 0; i < d_k; i++) atomicAdd(&acc[i], e * V[j * d_model + off + i]);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int i = tid; i < d_k; i += blockDim.x) {
        output[row * d_model + off + i] = acc[i] / row_sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {
    int d_k = d_model / h;
    dim3 grid(N, h);
    size_t smem = 2 * d_k * sizeof(float);
    mha_row_kernel<<<grid, THREADS, smem>>>(Q, K, V, output, N, d_model, h);
    cudaDeviceSynchronize();
}
