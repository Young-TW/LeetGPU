#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// One thread per token row: repeated max scan over E logits (E <= 256),
// tracking already-chosen experts in a 256-bit mask.
__global__ void topk_gating_kernel(const float* logits, float* topk_weights, int* topk_indices,
                                   int M, int E, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float* l = logits + (long long)row * E;
    unsigned long long chosen[4] = {0, 0, 0, 0};

    for (int t = 0; t < k; t++) {
        float best = -FLT_MAX;
        int best_e = -1;
        for (int e = 0; e < E; e++) {
            if (chosen[e >> 6] & (1ull << (e & 63))) continue;
            if (l[e] > best) {
                best = l[e];
                best_e = e;
            }
        }
        chosen[best_e >> 6] |= 1ull << (best_e & 63);
        topk_indices[(long long)row * k + t] = best_e;
    }

    // softmax over the selected logits (already in descending order, so the
    // first one is the max)
    float max_val = l[topk_indices[(long long)row * k]];
    float sum = 0.0f;
    for (int t = 0; t < k; t++) {
        sum += expf(l[topk_indices[(long long)row * k + t]] - max_val);
    }
    for (int t = 0; t < k; t++) {
        topk_weights[(long long)row * k + t] =
            expf(l[topk_indices[(long long)row * k + t]] - max_val) / sum;
    }
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    topk_gating_kernel<<<blocksPerGrid, threadsPerBlock>>>(logits, topk_weights, topk_indices, M,
                                                           E, k);
    cudaDeviceSynchronize();
}
