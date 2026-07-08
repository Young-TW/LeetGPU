#include <cuda_runtime.h>

// Interior: average of the four cardinal neighbors. Boundary rows/columns are copied.
__global__ void jacobi_kernel(const float* input, float* output, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;

    long long idx = (long long)r * cols + c;
    if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1) {
        output[idx] = input[idx];
    } else {
        output[idx] = 0.25f * (input[idx - cols] + input[idx + cols] + input[idx - 1] +
                               input[idx + 1]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    jacobi_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
