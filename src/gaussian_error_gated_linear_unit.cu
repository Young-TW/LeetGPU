#include <cuda_runtime.h>
#include <math.h>

// output[i] = x1[i] * GELU(x2[i]), GELU(x) = 0.5 x (1 + erf(x / sqrt(2)))
__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < halfN) {
        float x1 = input[i];
        float x2 = input[i + halfN];
        float gelu = 0.5f * x2 * (1.0f + erff(x2 * 0.70710678118654752f));
        output[i] = x1 * gelu;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
