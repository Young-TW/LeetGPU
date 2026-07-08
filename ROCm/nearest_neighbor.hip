#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <float.h>

#define TILE 256

// points[3*i .. 3*i+2] = (x, y, z); brute-force search tiled through shared memory.
__global__ void nn_kernel(const float* points, int* indices, int N) {
    __shared__ float tile[TILE * 3];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 0, y = 0, z = 0;
    if (i < N) {
        x = points[3 * i];
        y = points[3 * i + 1];
        z = points[3 * i + 2];
    }

    float best = FLT_MAX;
    int best_j = -1;

    for (int base = 0; base < N; base += TILE) {
        int count = (N - base < TILE) ? (N - base) : TILE;
        for (int t = threadIdx.x; t < count * 3; t += blockDim.x) {
            tile[t] = points[base * 3 + t];
        }
        __syncthreads();

        if (i < N) {
            for (int t = 0; t < count; t++) {
                int j = base + t;
                if (j == i) continue;
                float dx = x - tile[3 * t];
                float dy = y - tile[3 * t + 1];
                float dz = z - tile[3 * t + 2];
                float d = dx * dx + dy * dy + dz * dz;
                if (d < best) {
                    best = d;
                    best_j = j;
                }
            }
        }
        __syncthreads();
    }

    if (i < N) indices[i] = best_j;
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    int threadsPerBlock = TILE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    nn_kernel<<<blocksPerGrid, threadsPerBlock>>>(points, indices, N);
    hipDeviceSynchronize();
}
