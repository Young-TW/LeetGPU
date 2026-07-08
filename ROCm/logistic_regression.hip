#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <math.h>

// Newton's method (IRLS): beta += (X^T W X)^{-1} X^T (y - p), p = sigmoid(X beta).

#define NEWTON_ITERS 25

__global__ void predict_kernel(const float* X, const float* beta, float* p, float* w, int n,
                               int f) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    float z = 0.0f;
    for (int i = 0; i < f; i++) z += X[(long long)s * f + i] * beta[i];
    float pi = 1.0f / (1.0f + expf(-z));
    p[s] = pi;
    w[s] = pi * (1.0f - pi);
}

__global__ void gradient_kernel(const float* X, const float* y, const float* p, float* grad,
                                int n, int f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f) return;
    float sum = 0.0f;
    for (int s = 0; s < n; s++) sum += X[(long long)s * f + i] * (y[s] - p[s]);
    grad[i] = sum;
}

__global__ void hessian_kernel(const float* X, const float* w, float* H, int n, int f) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= f || j >= f || j < i) return;

    float sum = 0.0f;
    for (int s = 0; s < n; s++) {
        sum += w[s] * X[(long long)s * f + i] * X[(long long)s * f + j];
    }
    if (i == j) sum += 1e-6f;  // ridge for numerical stability
    H[i * f + j] = sum;
    H[j * f + i] = sum;
}

__global__ void update_kernel(float* beta, const float* delta, int f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < f) beta[i] += delta[i];
}

// Single-block SPD solve: in-place Cholesky (lower), then two triangular solves.
__global__ void cholesky_solve_kernel(float* A, const float* b, float* x, int f) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int nt = blockDim.x;

    for (int j = 0; j < f; j++) {
        float part = 0.0f;
        for (int k = tid; k < j; k += nt) part += A[j * f + k] * A[j * f + k];
        sdata[tid] = part;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) A[j * f + j] = sqrtf(fmaxf(A[j * f + j] - sdata[0], 1e-12f));
        __syncthreads();
        float diag = A[j * f + j];

        for (int i = j + 1 + tid; i < f; i += nt) {
            float sum = A[i * f + j];
            for (int k = 0; k < j; k++) sum -= A[i * f + k] * A[j * f + k];
            A[i * f + j] = sum / diag;
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int i = 0; i < f; i++) {
            float sum = b[i];
            for (int k = 0; k < i; k++) sum -= A[i * f + k] * x[k];
            x[i] = sum / A[i * f + i];
        }
        for (int i = f - 1; i >= 0; i--) {
            float sum = x[i];
            for (int k = i + 1; k < f; k++) sum -= A[k * f + i] * x[k];
            x[i] = sum / A[i * f + i];
        }
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    int n = n_samples, f = n_features;
    float *p, *w, *grad, *H, *delta;
    hipMalloc(&p, n * sizeof(float));
    hipMalloc(&w, n * sizeof(float));
    hipMalloc(&grad, f * sizeof(float));
    hipMalloc(&H, (size_t)f * f * sizeof(float));
    hipMalloc(&delta, f * sizeof(float));

    hipMemset(beta, 0, f * sizeof(float));

    dim3 t2(16, 16);
    dim3 g2((f + 15) / 16, (f + 15) / 16);
    int threads = 256;

    for (int it = 0; it < NEWTON_ITERS; it++) {
        predict_kernel<<<(n + threads - 1) / threads, threads>>>(X, beta, p, w, n, f);
        gradient_kernel<<<(f + threads - 1) / threads, threads>>>(X, y, p, grad, n, f);
        hessian_kernel<<<g2, t2>>>(X, w, H, n, f);
        cholesky_solve_kernel<<<1, threads, threads * sizeof(float)>>>(H, grad, delta, f);
        update_kernel<<<(f + threads - 1) / threads, threads>>>(beta, delta, f);
    }
    hipDeviceSynchronize();

    hipFree(p);
    hipFree(w);
    hipFree(grad);
    hipFree(H);
    hipFree(delta);
}
