#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <float.h>

__global__ void assign_kernel(const float* data_x, const float* data_y, int* labels,
                              const float* cx, const float* cy, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = data_x[i], y = data_y[i];
    float best = FLT_MAX;
    int best_c = 0;
    for (int c = 0; c < k; c++) {
        float dx = x - cx[c];
        float dy = y - cy[c];
        float d = dx * dx + dy * dy;
        if (d < best) {
            best = d;
            best_c = c;
        }
    }
    labels[i] = best_c;
}

__global__ void accumulate_kernel(const float* data_x, const float* data_y, const int* labels,
                                  float* sum_x, float* sum_y, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = labels[i];
    atomicAdd(&sum_x[c], data_x[i]);
    atomicAdd(&sum_y[c], data_y[i]);
    atomicAdd(&count[c], 1);
}

__global__ void update_kernel(float* cx, float* cy, const float* sum_x, const float* sum_y,
                              const int* count, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    if (count[c] > 0) {  // empty clusters keep their previous centroid
        cx[c] = sum_x[c] / count[c];
        cy[c] = sum_y[c] / count[c];
    }
}

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {
    hipMemcpy(final_centroid_x, initial_centroid_x, k * sizeof(float), hipMemcpyDeviceToDevice);
    hipMemcpy(final_centroid_y, initial_centroid_y, k * sizeof(float), hipMemcpyDeviceToDevice);

    float *sum_x, *sum_y;
    int* count;
    hipMalloc(&sum_x, k * sizeof(float));
    hipMalloc(&sum_y, k * sizeof(float));
    hipMalloc(&count, k * sizeof(int));

    int threads = 256;
    int blocks = (sample_size + threads - 1) / threads;
    int cblocks = (k + threads - 1) / threads;

    for (int it = 0; it < max_iterations; it++) {
        assign_kernel<<<blocks, threads>>>(data_x, data_y, labels, final_centroid_x,
                                           final_centroid_y, sample_size, k);
        hipMemset(sum_x, 0, k * sizeof(float));
        hipMemset(sum_y, 0, k * sizeof(float));
        hipMemset(count, 0, k * sizeof(int));
        accumulate_kernel<<<blocks, threads>>>(data_x, data_y, labels, sum_x, sum_y, count,
                                               sample_size);
        update_kernel<<<cblocks, threads>>>(final_centroid_x, final_centroid_y, sum_x, sum_y,
                                            count, k);
    }
    // final assignment against the converged centroids
    assign_kernel<<<blocks, threads>>>(data_x, data_y, labels, final_centroid_x, final_centroid_y,
                                       sample_size, k);
    hipDeviceSynchronize();

    hipFree(sum_x);
    hipFree(sum_y);
    hipFree(count);
}
