#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

// Same-size convolution with a centered odd kernel and zero padding.
__global__ void gaussian_blur_kernel(const float* input, const float* kernel, float* output,
                                     int input_rows, int input_cols, int kernel_rows,
                                     int kernel_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= input_rows || col >= input_cols) return;

    int kh2 = kernel_rows / 2;
    int kw2 = kernel_cols / 2;

    float sum = 0.0f;
    for (int m = -kh2; m <= kh2; m++) {
        int r = row + m;
        if (r < 0 || r >= input_rows) continue;
        for (int n = -kw2; n <= kw2; n++) {
            int c = col + n;
            if (c < 0 || c >= input_cols) continue;
            sum += input[r * input_cols + c] * kernel[(m + kh2) * kernel_cols + (n + kw2)];
        }
    }
    output[row * input_cols + col] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((input_cols + 15) / 16, (input_rows + 15) / 16);

    gaussian_blur_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows,
                                                             input_cols, kernel_rows, kernel_cols);
    hipDeviceSynchronize();
}
