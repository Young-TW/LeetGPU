#include <cuda_runtime.h>
#include <float.h>

// Bitonic sort on a power-of-two padded copy (padding = +inf so it sinks to the end).

__global__ void pad_kernel(const float* data, float* buf, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) buf[i] = (i < N) ? data[i] : FLT_MAX;
}

__global__ void bitonic_step_kernel(float* buf, int M, int j, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    int ixj = i ^ j;
    if (ixj > i) {
        bool ascending = ((i & k) == 0);
        float a = buf[i];
        float b = buf[ixj];
        if ((a > b) == ascending) {
            buf[i] = b;
            buf[ixj] = a;
        }
    }
}

__global__ void copy_back_kernel(const float* buf, float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] = buf[i];
}

// data is device pointer
extern "C" void solve(float* data, int N) {
    int M = 1;
    while (M < N) M <<= 1;

    float* buf;
    cudaMalloc(&buf, M * sizeof(float));

    int threads = 256;
    int blocksM = (M + threads - 1) / threads;
    pad_kernel<<<blocksM, threads>>>(data, buf, N, M);

    for (int k = 2; k <= M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_step_kernel<<<blocksM, threads>>>(buf, M, j, k);
        }
    }

    copy_back_kernel<<<(N + threads - 1) / threads, threads>>>(buf, data, N);
    cudaDeviceSynchronize();
    cudaFree(buf);
}
