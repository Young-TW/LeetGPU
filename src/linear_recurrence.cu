#include <cuda_runtime.h>

// h[b, 0] = x[b, 0]; h[b, t] = a[b, t] * h[b, t-1] + x[b, t]. One thread per batch row.
__global__ void recurrence_kernel(const float* a, const float* x, float* h, int B, int L) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* ab = a + (long long)b * L;
    const float* xb = x + (long long)b * L;
    float* hb = h + (long long)b * L;

    float state = xb[0];
    hb[0] = state;
    for (int t = 1; t < L; t++) {
        state = ab[t] * state + xb[t];
        hb[t] = state;
    }
}

// a, x, h are device pointers
extern "C" void solve(const float* a, const float* x, float* h, int B, int L) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (B + threadsPerBlock - 1) / threadsPerBlock;

    recurrence_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, x, h, B, L);
    cudaDeviceSynchronize();
}
