#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#define THREADS 128
#define MAX_D 256

// Decode-phase attention: Q is one token per head; K/V caches are int8 with
// per-token scales. K_float[h, s, d] = K_int8[h, s, d] * k_scale[h, s].
__global__ void int8_kv_attention_kernel(const float* Q, const int8_t* K_int8,
                                         const int8_t* V_int8, const float* k_scale,
                                         const float* v_scale, float* output, int seq_len,
                                         int head_dim) {
    __shared__ float q[MAX_D];
    __shared__ float acc[MAX_D];
    __shared__ float sdata[THREADS];

    int h = blockIdx.x;
    int tid = threadIdx.x;

    const int8_t* Kh = K_int8 + (long long)h * seq_len * head_dim;
    const int8_t* Vh = V_int8 + (long long)h * seq_len * head_dim;
    const float* ks = k_scale + (long long)h * seq_len;
    const float* vs = v_scale + (long long)h * seq_len;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        q[i] = Q[(long long)h * head_dim + i];
        acc[i] = 0.0f;
    }
    __syncthreads();

    float scale = (1.0f / sqrtf((float)head_dim));

    float local_max = -FLT_MAX;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < head_dim; i++) s += q[i] * (float)Kh[(long long)j * head_dim + i];
        s *= ks[j] * scale;
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
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < head_dim; i++) s += q[i] * (float)Kh[(long long)j * head_dim + i];
        s *= ks[j] * scale;
        float e = expf(s - row_max);
        local_sum += e;
        float vscale = e * vs[j];
        for (int i = 0; i < head_dim; i++) {
            atomicAdd(&acc[i], vscale * (float)Vh[(long long)j * head_dim + i]);
        }
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int i = tid; i < head_dim; i += blockDim.x) {
        output[(long long)h * head_dim + i] = acc[i] / row_sum;
    }
}

// Q, K_int8, V_int8, k_scale, v_scale, output are device pointers
extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {
    int8_kv_attention_kernel<<<num_heads, THREADS>>>(Q, K_int8, V_int8, k_scale, v_scale, output,
                                                     seq_len, head_dim);
    hipDeviceSynchronize();
}
