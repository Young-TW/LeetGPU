#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int local_hist[];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) local_hist[i] = 0;
    __syncthreads();

    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (long long)gridDim.x * blockDim.x) {
        atomicAdd(&local_hist[input[i]], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (local_hist[i] > 0) atomicAdd(&histogram[i], local_hist[i]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 2048) blocksPerGrid = 2048;

    cudaMemset(histogram, 0, num_bins * sizeof(int));
    histogram_kernel<<<blocksPerGrid, threadsPerBlock, num_bins * sizeof(int)>>>(input, histogram,
                                                                                 N, num_bins);
    cudaDeviceSynchronize();
}
