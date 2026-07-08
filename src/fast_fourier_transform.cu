#include <cuda_runtime.h>
#include <math.h>

// Iterative radix-2 Cooley-Tukey for power-of-two N; direct DFT fallback otherwise.
// Complex values are interleaved: [re0, im0, re1, im1, ...].

__device__ inline unsigned int bit_reverse32(unsigned int x) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
    return (x << 16) | (x >> 16);
}

__global__ void bit_reverse_kernel(const float* in, float* out, int N, int log2N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned int r = bit_reverse32((unsigned int)i) >> (32 - log2N);
    out[2 * r] = in[2 * i];
    out[2 * r + 1] = in[2 * i + 1];
}

__global__ void butterfly_kernel(float* data, int N, int m) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;  // butterfly index
    if (t >= N / 2) return;
    int half = m / 2;
    int group = t / half;
    int pos = t % half;
    int base = group * m + pos;

    float angle = -2.0f * (float)M_PI * pos / m;
    float wr = cosf(angle);
    float wi = sinf(angle);

    float ar = data[2 * base], ai = data[2 * base + 1];
    float br = data[2 * (base + half)], bi = data[2 * (base + half) + 1];
    float tr = br * wr - bi * wi;
    float ti = br * wi + bi * wr;

    data[2 * base] = ar + tr;
    data[2 * base + 1] = ai + ti;
    data[2 * (base + half)] = ar - tr;
    data[2 * (base + half) + 1] = ai - ti;
}

__global__ void naive_dft_kernel(const float* signal, float* spectrum, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    float re = 0.0f, im = 0.0f;
    for (int n = 0; n < N; n++) {
        float angle = -2.0 * M_PI * (double)k * n / N;
        float c = cosf(angle), s = sinf(angle);
        float xr = signal[2 * n], xi = signal[2 * n + 1];
        re += xr * c - xi * s;
        im += xr * s + xi * c;
    }
    spectrum[2 * k] = re;
    spectrum[2 * k + 1] = im;
}

// signal and spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int N) {
    int threads = 256;

    if ((N & (N - 1)) != 0) {  // not a power of two
        naive_dft_kernel<<<(N + threads - 1) / threads, threads>>>(signal, spectrum, N);
        cudaDeviceSynchronize();
        return;
    }

    int log2N = 0;
    while ((1 << log2N) < N) log2N++;

    int blocksN = (N + threads - 1) / threads;
    bit_reverse_kernel<<<blocksN, threads>>>(signal, spectrum, N, log2N);

    int blocksHalf = (N / 2 + threads - 1) / threads;
    if (blocksHalf == 0) blocksHalf = 1;
    for (int m = 2; m <= N; m <<= 1) {
        butterfly_kernel<<<blocksHalf, threads>>>(spectrum, N, m);
    }
    cudaDeviceSynchronize();
}
