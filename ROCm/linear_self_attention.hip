#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <math.h>

// Linear attention: out_i = phi(Q_i) (phi(K)^T V) / (phi(Q_i) . sum_j phi(K_j)),
// phi(x) = ELU(x) + 1.

__device__ inline float phi(float x) {
    return x > 0.0f ? x + 1.0f : expf(x);
}

// Accumulate S = phi(K)^T V (d x d) and z = sum_j phi(K_j) (d).
__global__ void kv_accumulate_kernel(const float* K, const float* V, float* S, float* z, int M,
                                     int d) {
    int j = blockIdx.x;  // key/value row
    for (int t = threadIdx.x; t < d * d; t += blockDim.x) {
        int a = t / d;  // feature index of K
        int b = t % d;  // feature index of V
        atomicAdd(&S[a * d + b], phi(K[(long long)j * d + a]) * V[(long long)j * d + b]);
    }
    for (int a = threadIdx.x; a < d; a += blockDim.x) {
        atomicAdd(&z[a], phi(K[(long long)j * d + a]));
    }
}

__global__ void output_kernel(const float* Q, const float* S, const float* z, float* output,
                              int M, int d) {
    int i = blockIdx.x;
    extern __shared__ float pq[];  // phi(Q_i), plus denominator at pq[d]

    for (int a = threadIdx.x; a < d; a += blockDim.x) {
        pq[a] = phi(Q[(long long)i * d + a]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float denom = 0.0f;
        for (int a = 0; a < d; a++) denom += pq[a] * z[a];
        pq[d] = denom;
    }
    __syncthreads();

    float denom = pq[d];
    for (int b = threadIdx.x; b < d; b += blockDim.x) {
        float num = 0.0f;
        for (int a = 0; a < d; a++) num += pq[a] * S[a * d + b];
        output[(long long)i * d + b] = num / denom;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M,
                      int d) {
    float *S, *z;
    hipMalloc(&S, (size_t)d * d * sizeof(float));
    hipMalloc(&z, d * sizeof(float));
    hipMemset(S, 0, (size_t)d * d * sizeof(float));
    hipMemset(z, 0, d * sizeof(float));

    kv_accumulate_kernel<<<M, 128>>>(K, V, S, z, M, d);
    output_kernel<<<M, 128, (d + 1) * sizeof(float)>>>(Q, S, z, output, M, d);
    hipDeviceSynchronize();

    hipFree(S);
    hipFree(z);
}
