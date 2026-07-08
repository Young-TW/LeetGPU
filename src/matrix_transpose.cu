#include <cuda_runtime.h>

#define TILE 32

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];  // +1 avoids shared memory bank conflicts

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
    }
    __syncthreads();

    int out_col = blockIdx.y * TILE + threadIdx.x;  // index into rows dimension
    int out_row = blockIdx.x * TILE + threadIdx.y;  // index into cols dimension
    if (out_row < cols && out_col < rows) {
        output[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
