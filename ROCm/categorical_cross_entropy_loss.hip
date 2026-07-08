#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>

// One block per sample: loss_j = log(sum_k exp(z_jk)) - z_{j,y_j},
// computed stably as max_j + log(sum exp(z - max)) - z_{j,y_j}.
__global__ void ce_loss_kernel(const float* logits, const int* true_labels, float* loss, int N,
                               int C) {
    __shared__ float sdata[256];
    int j = blockIdx.x;
    int tid = threadIdx.x;
    const float* row = logits + (long long)j * C;

    float m = -FLT_MAX;
    for (int k = tid; k < C; k += blockDim.x) m = fmaxf(m, row[k]);
    sdata[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float sum = 0.0f;
    for (int k = tid; k < C; k += blockDim.x) sum += expf(row[k] - row_max);
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float loss_j = row_max + logf(sdata[0]) - row[true_labels[j]];
        atomicAdd(loss, loss_j / N);
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    hipMemset(loss, 0, sizeof(float));
    ce_loss_kernel<<<N, 256>>>(logits, true_labels, loss, N, C);
    hipDeviceSynchronize();
}
