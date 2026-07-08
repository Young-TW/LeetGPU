#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

// Level-synchronous BFS: expand the frontier one level per kernel launch.
__global__ void bfs_step_kernel(const int* grid, int* dist, int rows, int cols, int level,
                                int* changed) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    if (dist[r * cols + c] != level) return;

    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};
    for (int k = 0; k < 4; k++) {
        int nr = r + dr[k], nc = c + dc[k];
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
        int nidx = nr * cols + nc;
        if (grid[nidx] == 0 && dist[nidx] == -1) {
            dist[nidx] = level + 1;
            *changed = 1;
        }
    }
}

__global__ void init_dist_kernel(int* dist, long long total, int start_idx) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dist[i] = (i == start_idx) ? 0 : -1;
}

__global__ void write_result_kernel(const int* dist, int* result, int end_idx) {
    *result = dist[end_idx];
}

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row,
                      int start_col, int end_row, int end_col) {
    long long total = (long long)rows * cols;
    int* dist;
    int* changed;
    hipMalloc(&dist, total * sizeof(int));
    hipMalloc(&changed, sizeof(int));

    int start_idx = start_row * cols + start_col;
    int end_idx = end_row * cols + end_col;
    init_dist_kernel<<<(int)((total + 255) / 256), 256>>>(dist, total, start_idx);

    dim3 t2(16, 16);
    dim3 g2((cols + 15) / 16, (rows + 15) / 16);

    int h_changed = 1;
    int h_found = 0;
    for (int level = 0; h_changed && !h_found; level++) {
        hipMemset(changed, 0, sizeof(int));
        bfs_step_kernel<<<g2, t2>>>(grid, dist, rows, cols, level, changed);
        hipMemcpy(&h_changed, changed, sizeof(int), hipMemcpyDeviceToHost);
        hipMemcpy(&h_found, dist + end_idx, sizeof(int), hipMemcpyDeviceToHost);
        h_found = (h_found != -1);
    }

    write_result_kernel<<<1, 1>>>(dist, result, end_idx);
    hipDeviceSynchronize();
    hipFree(dist);
    hipFree(changed);
}
