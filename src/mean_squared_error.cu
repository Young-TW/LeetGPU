#include <cuda_runtime.h>

__global__ void mse_sum_kernel(const float* predictions, const float* targets, float* mse, int N) {
    __shared__ float sdata[256];

    float sum = 0.0f;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (long long)gridDim.x * blockDim.x) {
        float d = predictions[i] - targets[i];
        sum += d * d;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(mse, sdata[0]);
}

__global__ void divide_kernel(float* mse, int N) {
    *mse /= N;
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 4096) blocksPerGrid = 4096;

    cudaMemset(mse, 0, sizeof(float));
    mse_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, mse, N);
    divide_kernel<<<1, 1>>>(mse, N);
    cudaDeviceSynchronize();
}
