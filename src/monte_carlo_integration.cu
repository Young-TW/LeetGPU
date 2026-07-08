#include <cuda_runtime.h>

__global__ void sum_kernel(const float* y_samples, float* result, int n) {
    __shared__ float sdata[256];

    float sum = 0.0f;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        sum += y_samples[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(result, sdata[0]);
}

__global__ void finalize_kernel(float* result, float a, float b, int n) {
    *result = (b - a) * (*result) / n;
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_samples + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 4096) blocksPerGrid = 4096;

    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(y_samples, result, n_samples);
    finalize_kernel<<<1, 1>>>(result, a, b, n_samples);
    cudaDeviceSynchronize();
}
