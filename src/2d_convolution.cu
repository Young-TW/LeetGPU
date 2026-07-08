#include <cuda_runtime.h>

__global__ void convolution_2d_kernel(const float* input, const float* kernel, float* output,
                                      int input_rows, int input_cols, int kernel_rows,
                                      int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < output_rows && col < output_cols) {
        float sum = 0.0f;
        for (int m = 0; m < kernel_rows; m++) {
            for (int n = 0; n < kernel_cols; n++) {
                sum += input[(row + m) * input_cols + (col + n)] * kernel[m * kernel_cols + n];
            }
        }
        output[row * output_cols + col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((output_cols + 15) / 16, (output_rows + 15) / 16);

    convolution_2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows,
                                                              input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
