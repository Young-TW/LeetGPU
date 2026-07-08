#include <cuda_runtime.h>

#define RADIUS 5.0f
#define ALPHA 0.05f

__global__ void flock_kernel(const float* agents, float* agents_next, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = agents[4 * i + 0];
    float y = agents[4 * i + 1];
    float vx = agents[4 * i + 2];
    float vy = agents[4 * i + 3];

    float sum_vx = 0.0f, sum_vy = 0.0f;
    int count = 0;
    float r2 = RADIUS * RADIUS;
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        float dx = x - agents[4 * j + 0];
        float dy = y - agents[4 * j + 1];
        if (dx * dx + dy * dy < r2) {
            sum_vx += agents[4 * j + 2];
            sum_vy += agents[4 * j + 3];
            count++;
        }
    }

    float avg_vx = count > 0 ? sum_vx / count : vx;
    float avg_vy = count > 0 ? sum_vy / count : vy;

    float new_vx = vx + ALPHA * (avg_vx - vx);
    float new_vy = vy + ALPHA * (avg_vy - vy);

    agents_next[4 * i + 0] = x + new_vx;
    agents_next[4 * i + 1] = y + new_vy;
    agents_next[4 * i + 2] = new_vx;
    agents_next[4 * i + 3] = new_vy;
}

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    flock_kernel<<<blocksPerGrid, threadsPerBlock>>>(agents, agents_next, N);
    cudaDeviceSynchronize();
}
