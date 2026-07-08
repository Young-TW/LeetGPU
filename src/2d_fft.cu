#include <cuda_runtime.h>
#include <math.h>

// 2D DFT by row-column decomposition: batched 1D FFT along rows, transpose,
// FFT along rows again, transpose back. Radix-2 for power-of-two lengths,
// naive DFT otherwise. Data is interleaved complex, row-major.

__device__ inline unsigned int bit_reverse32(unsigned int x) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
    return (x << 16) | (x >> 16);
}

__global__ void bit_reverse_rows_kernel(const float* in, float* out, int rows, int n, int log2n) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)rows * n) return;
    int r = t / n;
    int i = t % n;
    unsigned int rev = bit_reverse32((unsigned int)i) >> (32 - log2n);
    long long src = 2 * ((long long)r * n + i);
    long long dst = 2 * ((long long)r * n + rev);
    out[dst] = in[src];
    out[dst + 1] = in[src + 1];
}

__global__ void butterfly_rows_kernel(float* data, int rows, int n, int m) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long per_row = n / 2;
    if (t >= (long long)rows * per_row) return;
    int r = t / per_row;
    int u = t % per_row;
    int half = m / 2;
    int group = u / half;
    int pos = u % half;
    long long base = (long long)r * n + group * m + pos;

    float angle = -2.0f * (float)M_PI * pos / m;
    float wr = cosf(angle), wi = sinf(angle);

    float ar = data[2 * base], ai = data[2 * base + 1];
    float br = data[2 * (base + half)], bi = data[2 * (base + half) + 1];
    float tr = br * wr - bi * wi;
    float ti = br * wi + bi * wr;

    data[2 * base] = ar + tr;
    data[2 * base + 1] = ai + ti;
    data[2 * (base + half)] = ar - tr;
    data[2 * (base + half) + 1] = ai - ti;
}

__global__ void naive_dft_rows_kernel(const float* in, float* out, int rows, int n) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)rows * n) return;
    int r = t / n;
    int k = t % n;

    float re = 0.0f, im = 0.0f;
    for (int j = 0; j < n; j++) {
        float angle = -2.0 * M_PI * (double)k * j / n;
        float c = cosf(angle), s = sinf(angle);
        float xr = in[2 * ((long long)r * n + j)];
        float xi = in[2 * ((long long)r * n + j) + 1];
        re += xr * c - xi * s;
        im += xr * s + xi * c;
    }
    out[2 * ((long long)r * n + k)] = re;
    out[2 * ((long long)r * n + k) + 1] = im;
}

// Transpose an rows x cols complex matrix.
__global__ void transpose_complex_kernel(const float* in, float* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    long long src = 2 * ((long long)r * cols + c);
    long long dst = 2 * ((long long)c * rows + r);
    out[dst] = in[src];
    out[dst + 1] = in[src + 1];
}

// FFT each of `rows` rows of length n; input in `data`, result in `data` (uses tmp).
static void fft_rows(float* data, float* tmp, int rows, int n) {
    int threads = 256;
    long long total = (long long)rows * n;
    int blocks = (int)((total + threads - 1) / threads);

    if ((n & (n - 1)) != 0) {
        naive_dft_rows_kernel<<<blocks, threads>>>(data, tmp, rows, n);
        cudaMemcpy(data, tmp, total * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }

    int log2n = 0;
    while ((1 << log2n) < n) log2n++;

    bit_reverse_rows_kernel<<<blocks, threads>>>(data, tmp, rows, n, log2n);
    cudaMemcpy(data, tmp, total * 2 * sizeof(float), cudaMemcpyDeviceToDevice);

    long long bt = (long long)rows * (n / 2);
    int bblocks = (int)((bt + threads - 1) / threads);
    if (bblocks == 0) bblocks = 1;
    for (int m = 2; m <= n; m <<= 1) {
        butterfly_rows_kernel<<<bblocks, threads>>>(data, rows, n, m);
    }
}

// signal, spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int M, int N) {
    long long total = (long long)M * N;
    float* tmp;
    cudaMalloc(&tmp, total * 2 * sizeof(float));
    cudaMemcpy(spectrum, signal, total * 2 * sizeof(float), cudaMemcpyDeviceToDevice);

    // FFT along rows (length N)
    fft_rows(spectrum, tmp, M, N);

    // transpose to N x M, FFT along rows (length M), transpose back
    dim3 t2(16, 16);
    dim3 g1((N + 15) / 16, (M + 15) / 16);
    transpose_complex_kernel<<<g1, t2>>>(spectrum, tmp, M, N);
    cudaMemcpy(spectrum, tmp, total * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
    fft_rows(spectrum, tmp, N, M);
    dim3 g2((M + 15) / 16, (N + 15) / 16);
    transpose_complex_kernel<<<g2, t2>>>(spectrum, tmp, N, M);
    cudaMemcpy(spectrum, tmp, total * 2 * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();
    cudaFree(tmp);
}
