#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define THREADS 128
#define MAX_D 256

// Layout: Q[h, s, d] with num_q_heads heads; K/V with num_kv_heads heads.
// Query head qh attends to kv head qh / (num_q_heads / num_kv_heads).
__global__ void gqa_kernel(const float* Q, const float* K, const float* V, float* output,
                           int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    __shared__ float q[MAX_D];
    __shared__ float acc[MAX_D];
    __shared__ float sdata[THREADS];

    int s_idx = blockIdx.x;
    int qh = blockIdx.y;
    int kvh = qh / (num_q_heads / num_kv_heads);
    int tid = threadIdx.x;

    const float* qrow = Q + ((long long)qh * seq_len + s_idx) * head_dim;
    const float* Kh = K + (long long)kvh * seq_len * head_dim;
    const float* Vh = V + (long long)kvh * seq_len * head_dim;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        q[i] = qrow[i];
        acc[i] = 0.0f;
    }
    __syncthreads();

    float scale = (1.0f / sqrtf((float)head_dim));

    float local_max = -FLT_MAX;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < head_dim; i++) s += q[i] * Kh[(long long)j * head_dim + i];
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
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < head_dim; i++) s += q[i] * Kh[(long long)j * head_dim + i];
        float e = expf(s * scale - row_max);
        local_sum += e;
        for (int i = 0; i < head_dim; i++) atomicAdd(&acc[i], e * Vh[(long long)j * head_dim + i]);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    float* out = output + ((long long)qh * seq_len + s_idx) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        out[i] = acc[i] / row_sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    dim3 grid(seq_len, num_q_heads);
    gqa_kernel<<<grid, THREADS>>>(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim);
    cudaDeviceSynchronize();
}
