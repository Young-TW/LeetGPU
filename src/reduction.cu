#include <cuda_runtime.h>

__global__ void reduce_kernel(const float* input, float* output, int N) {
    __shared__ float sdata[256];

    float sum = 0.0f;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (long long)gridDim.x * blockDim.x) {
        sum += input[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 4096) blocksPerGrid = 4096;

    cudaMemset(output, 0, sizeof(float));
    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
