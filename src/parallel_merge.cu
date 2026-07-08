#include <cuda_runtime.h>

// Each thread places one output element via merge-path binary search:
// output position i takes the k-th smallest of A ∪ B, found by searching the
// split point a (elements taken from A) on the diagonal a + b = i.
__global__ void merge_kernel(const float* A, const float* B, float* C, int M, int N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M + N) return;

    long long lo = i > N ? i - N : 0;
    long long hi = i < M ? i : M;
    while (lo < hi) {
        long long a = (lo + hi) / 2;
        long long b = i - a;
        // taking a elements from A is feasible iff A[a] >= B[b-1]
        if (b > 0 && a < M && A[a] < B[b - 1]) {
            lo = a + 1;
        } else {
            hi = a;
        }
    }
    long long a = lo;
    long long b = i - a;

    float va = (a < M) ? A[a] : 0.0f;
    float vb = (b < N) ? B[b] : 0.0f;
    if (a < M && (b >= N || va <= vb)) {
        C[i] = va;
    } else {
        C[i] = vb;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    long long total = (long long)M + N;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;

    merge_kernel<<<(int)blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
