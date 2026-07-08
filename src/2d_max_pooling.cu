#include <cuda_runtime.h>
#include <float.h>

// output[n, c, h_out, w_out] = max over the kernel window; padded cells are
// ignored (window max only considers valid input positions).
__global__ void max_pool_kernel(const float* input, float* output, int N, int C, int H, int W,
                                int kernel_size, int stride, int padding, int H_out, int W_out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / ((long long)W_out * H_out)) % C;
    int n = idx / ((long long)W_out * H_out * C);

    int h0 = h_out * stride - padding;
    int w0 = w_out * stride - padding;

    const float* plane = input + ((long long)n * C + c) * H * W;
    float m = -FLT_MAX;
    for (int i = 0; i < kernel_size; i++) {
        int h = h0 + i;
        if (h < 0 || h >= H) continue;
        for (int j = 0; j < kernel_size; j++) {
            int w = w0 + j;
            if (w < 0 || w >= W) continue;
            m = fmaxf(m, plane[h * W + w]);
        }
    }
    output[idx] = m;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    long long total = (long long)N * C * H_out * W_out;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;

    max_pool_kernel<<<(int)blocks, threads>>>(input, output, N, C, H, W, kernel_size, stride,
                                              padding, H_out, W_out);
    cudaDeviceSynchronize();
}
