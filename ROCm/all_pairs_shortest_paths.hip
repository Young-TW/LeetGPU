#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

// Floyd-Warshall: one kernel launch per intermediate vertex k.
__global__ void fw_step_kernel(float* output, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;

    float through_k = output[(long long)i * N + k] + output[(long long)k * N + j];
    long long idx = (long long)i * N + j;
    if (through_k < output[idx]) {
        output[idx] = through_k;
    }
}

// dist, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* dist, float* output, int N) {
    hipMemcpy(output, dist, (size_t)N * N * sizeof(float), hipMemcpyDeviceToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    for (int k = 0; k < N; k++) {
        fw_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, N, k);
    }
    hipDeviceSynchronize();
}
