#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

// output[b, l, d] = bias[d] + sum_k weight[d, k] * x[b, l-k, d], zero-padded on the left.
// Channels-last layout: x[b, l, d] at b*L*D + l*D + d.
__global__ void causal_dwconv_kernel(const float* x, const float* weight, const float* bias,
                                     float* output, int B, int L, int D, int K) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * L * D;
    if (idx >= total) return;

    int d = idx % D;
    int l = (idx / D) % L;
    int b = idx / ((long long)D * L);

    float sum = bias[d];
    for (int k = 0; k < K; k++) {
        int src = l - k;
        if (src < 0) break;
        sum += weight[d * K + k] * x[((long long)b * L + src) * D + d];
    }
    output[idx] = sum;
}

// x, weight, bias, output are device pointers
extern "C" void solve(const float* x, const float* weight, const float* bias, float* output, int B,
                      int L, int D, int K) {
    long long total = (long long)B * L * D;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;

    causal_dwconv_kernel<<<(int)blocks, threads>>>(x, weight, bias, output, B, L, D, K);
    hipDeviceSynchronize();
}
