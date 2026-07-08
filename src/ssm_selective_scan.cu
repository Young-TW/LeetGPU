#include <cuda_runtime.h>
#include <math.h>

#define MAX_STATE 64

// One thread per (batch, channel): sequential scan over time with the per-channel
// hidden state h[d_state] kept in registers/local memory.
//   A_bar = exp(delta * A[d, n]);  B_bar = delta * B[b, t, n]
//   h[n] = A_bar * h[n] + B_bar * u;  y = sum_n C[b, t, n] * h[n] + skip[d] * u
__global__ void selective_scan_kernel(const float* u, const float* delta, const float* A,
                                      const float* B, const float* C, const float* skip, float* y,
                                      int batch, int seq_len, int d_model, int d_state) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)batch * d_model) return;

    int d = idx % d_model;
    int b = idx / d_model;

    float h[MAX_STATE];
    for (int n = 0; n < d_state; n++) h[n] = 0.0f;

    const float* Ad = A + (long long)d * d_state;
    float sk = skip[d];

    for (int t = 0; t < seq_len; t++) {
        long long bt = (long long)b * seq_len + t;
        float ut = u[bt * d_model + d];
        float dt = delta[bt * d_model + d];
        const float* Bt = B + bt * d_state;
        const float* Ct = C + bt * d_state;

        float out = 0.0f;
        for (int n = 0; n < d_state; n++) {
            h[n] = expf(dt * Ad[n]) * h[n] + dt * Bt[n] * ut;
            out += Ct[n] * h[n];
        }
        y[bt * d_model + d] = out + sk * ut;
    }
}

// u, delta, A, B, C, skip, y are device pointers
extern "C" void solve(const float* u, const float* delta, const float* A, const float* B,
                      const float* C, const float* skip, float* y, int batch, int seq_len,
                      int d_model, int d_state) {
    long long total = (long long)batch * d_model;
    int threads = 128;
    long long blocks = (total + threads - 1) / threads;

    selective_scan_kernel<<<(int)blocks, threads>>>(u, delta, A, B, C, skip, y, batch, seq_len,
                                                    d_model, d_state);
    cudaDeviceSynchronize();
}
