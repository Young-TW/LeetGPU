#include <cuda_runtime.h>

__global__ void range_sum_kernel(const int* input, int* output, int S, int E) {
    __shared__ int sdata[256];

    int sum = 0;
    for (long long i = S + blockIdx.x * blockDim.x + threadIdx.x; i <= E;
         i += (long long)gridDim.x * blockDim.x) {
        sum += input[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && sdata[0] != 0) atomicAdd(output, sdata[0]);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    long long count = (long long)E - S + 1;
    int threadsPerBlock = 256;
    long long blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 4096) blocksPerGrid = 4096;

    cudaMemset(output, 0, sizeof(int));
    range_sum_kernel<<<(int)blocksPerGrid, threadsPerBlock>>>(input, output, S, E);
    cudaDeviceSynchronize();
}
