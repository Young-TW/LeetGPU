#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <math.h>

// One block per (batch, time): s = E_T[token] + E_P[position], then LayerNorm over D.
__global__ void token_embedding_kernel(const int* token_ids, const int* position_ids,
                                       const float* token_embeddings,
                                       const float* position_embeddings, const float* gamma,
                                       const float* beta, float* output, int T, int D, float eps) {
    extern __shared__ float smem[];  // s[D] + reduction scratch [blockDim.x]
    float* s = smem;
    float* sdata = smem + D;

    int b = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;

    int tok = token_ids[(long long)b * T + t];
    int pos = position_ids[t];
    const float* et = token_embeddings + (long long)tok * D;
    const float* ep = position_embeddings + (long long)pos * D;

    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = et[d] + ep[d];
        s[d] = v;
        sum += v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int r = blockDim.x / 2; r > 0; r >>= 1) {
        if (tid < r) sdata[tid] += sdata[tid + r];
        __syncthreads();
    }
    float mean = sdata[0] / D;
    __syncthreads();

    float var_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = s[d] - mean;
        var_sum += v * v;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int r = blockDim.x / 2; r > 0; r >>= 1) {
        if (tid < r) sdata[tid] += sdata[tid + r];
        __syncthreads();
    }
    float inv_std = (1.0f / sqrtf(sdata[0] / D + eps));
    __syncthreads();

    float* out = output + ((long long)b * T + t) * D;
    for (int d = tid; d < D; d += blockDim.x) {
        out[d] = gamma[d] * (s[d] - mean) * inv_std + beta[d];
    }
}

// token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device
// pointers
extern "C" void solve(const int* token_ids, const int* position_ids, const float* token_embeddings,
                      const float* position_embeddings, const float* gamma, const float* beta,
                      float* output, int B, int T, int V, int P, int D, float eps) {
    dim3 grid(B, T);
    int threads = 256;
    size_t smem = (D + threads) * sizeof(float);
    token_embedding_kernel<<<grid, threads, smem>>>(token_ids, position_ids, token_embeddings,
                                                    position_embeddings, gamma, beta, output, T, D,
                                                    eps);
    hipDeviceSynchronize();
}
