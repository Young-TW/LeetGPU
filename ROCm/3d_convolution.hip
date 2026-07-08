#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

__global__ void convolution_3d_kernel(const float* input, const float* kernel, float* output,
                                      int input_depth, int input_rows, int input_cols,
                                      int kernel_depth, int kernel_rows, int kernel_cols) {
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;
    if (dep < output_depth && row < output_rows && col < output_cols) {
        float sum = 0.0f;
        for (int d = 0; d < kernel_depth; d++) {
            for (int r = 0; r < kernel_rows; r++) {
                for (int c = 0; c < kernel_cols; c++) {
                    float in = input[(dep + d) * input_rows * input_cols + (row + r) * input_cols +
                                     (col + c)];
                    float k = kernel[d * kernel_rows * kernel_cols + r * kernel_cols + c];
                    sum += in * k;
                }
            }
        }
        output[dep * output_rows * output_cols + row * output_cols + col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid((output_cols + 7) / 8, (output_rows + 7) / 8, (output_depth + 7) / 8);

    convolution_3d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_depth,
                                                              input_rows, input_cols, kernel_depth,
                                                              kernel_rows, kernel_cols);
    hipDeviceSynchronize();
}
