#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <math.h>

// beta = (X^T X)^{-1} X^T y via normal equations + Cholesky (X^T X is SPD for full-rank X).

__global__ void xtx_kernel(const float* X, float* XtX, int n, int f) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= f || j >= f || j < i) return;  // fill upper triangle, mirror below

    float sum = 0.0f;
    for (int s = 0; s < n; s++) {
        sum += X[(long long)s * f + i] * X[(long long)s * f + j];
    }
    XtX[i * f + j] = sum;
    XtX[j * f + i] = sum;
}

__global__ void xty_kernel(const float* X, const float* y, float* Xty, int n, int f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f) return;
    float sum = 0.0f;
    for (int s = 0; s < n; s++) sum += X[(long long)s * f + i] * y[s];
    Xty[i] = sum;
}

// Single-block SPD solve: in-place Cholesky (lower), then two triangular solves.
__global__ void cholesky_solve_kernel(float* A, const float* b, float* x, int f) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int nt = blockDim.x;

    for (int j = 0; j < f; j++) {
        // diagonal: A[j][j] = sqrt(A[j][j] - sum_k A[j][k]^2), reduced in parallel
        float part = 0.0f;
        for (int k = tid; k < j; k += nt) part += A[j * f + k] * A[j * f + k];
        sdata[tid] = part;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) A[j * f + j] = sqrtf(A[j * f + j] - sdata[0]);
        __syncthreads();
        float diag = A[j * f + j];

        // column below the diagonal
        for (int i = j + 1 + tid; i < f; i += nt) {
            float sum = A[i * f + j];
            for (int k = 0; k < j; k++) sum -= A[i * f + k] * A[j * f + k];
            A[i * f + j] = sum / diag;
        }
        __syncthreads();
    }

    if (tid == 0) {
        // forward solve L z = b (z stored in x)
        for (int i = 0; i < f; i++) {
            float sum = b[i];
            for (int k = 0; k < i; k++) sum -= A[i * f + k] * x[k];
            x[i] = sum / A[i * f + i];
        }
        // backward solve L^T beta = z
        for (int i = f - 1; i >= 0; i--) {
            float sum = x[i];
            for (int k = i + 1; k < f; k++) sum -= A[k * f + i] * x[k];
            x[i] = sum / A[i * f + i];
        }
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    float *XtX, *Xty;
    hipMalloc(&XtX, (size_t)n_features * n_features * sizeof(float));
    hipMalloc(&Xty, n_features * sizeof(float));

    dim3 t2(16, 16);
    dim3 g2((n_features + 15) / 16, (n_features + 15) / 16);
    xtx_kernel<<<g2, t2>>>(X, XtX, n_samples, n_features);
    xty_kernel<<<(n_features + 255) / 256, 256>>>(X, y, Xty, n_samples, n_features);

    int threads = 256;
    cholesky_solve_kernel<<<1, threads, threads * sizeof(float)>>>(XtX, Xty, beta, n_features);
    hipDeviceSynchronize();

    hipFree(XtX);
    hipFree(Xty);
}
