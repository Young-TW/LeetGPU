#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

// out = Q * cos + rotate_half(Q) * sin, where
// rotate_half(x) = [-x[d/2:], x[:d/2]].
__global__ void rope_kernel(const float* Q, const float* cos_t, const float* sin_t, float* output,
                            long long total, int D) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int j = idx % D;
    long long row_base = idx - j;
    int half = D / 2;

    float rot = (j < half) ? -Q[row_base + j + half] : Q[row_base + j - half];
    output[idx] = Q[idx] * cos_t[idx] + rot * sin_t[idx];
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    long long total = (long long)M * D;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;

    rope_kernel<<<(int)blocks, threads>>>(Q, cos, sin, output, total, D);
    hipDeviceSynchronize();
}
