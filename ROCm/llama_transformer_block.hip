#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>

// Llama-style block: RMSNorm, GQA (8 Q heads / 2 KV heads, head_dim 64), RoPE,
// causal attention, SwiGLU FFN (512 -> 1408 -> 512). No biases. eps = 1e-5.
#define D_MODEL 512
#define N_Q_HEADS 8
#define N_KV_HEADS 2
#define D_HEAD 64
#define KV_DIM (N_KV_HEADS * D_HEAD)  // 128
#define FFN_DIM 1408
#define ROPE_HALF 32

// Weight offsets (all matrices row-major with shape (out_dim, in_dim))
#define OFF_W1 0
#define OFF_WQ 512
#define OFF_WK 262656
#define OFF_WV 328192
#define OFF_WO 393728
#define OFF_W2 655872
#define OFF_WGATE 656384
#define OFF_WUP 1377280
#define OFF_WDOWN 2098176

#define TILE 16

// RMSNorm over the last dimension; one block per row.
__global__ void rmsnorm_kernel(const float* in, float* out, const float* w, int T, int d) {
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = in + (long long)row * d;

    float sum = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) sum += x[i] * x[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = (1.0f / sqrtf(sdata[0] / d + 1e-5f));
    __syncthreads();

    for (int i = tid; i < d; i += blockDim.x) {
        out[(long long)row * d + i] = x[i] * inv_rms * w[i];
    }
}

// C[T x N] = A[T x K] @ W^T, with W stored (N, K) row-major.
__global__ void matmul_wt_kernel(const float* A, const float* W, float* C, int T, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Ws[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int w_col = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < T && a_col < K) ? A[(long long)row * K + a_col] : 0.0f;
        // Ws[i][j] holds W[col_block + j][k_block + i]
        int w_row = blockIdx.x * TILE + threadIdx.x;
        Ws[threadIdx.y][threadIdx.x] =
            (w_row < N && w_col < K) ? W[(long long)w_row * K + w_col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += As[threadIdx.y][i] * Ws[i][threadIdx.x];
        __syncthreads();
    }

    if (row < T && col < N) {
        C[(long long)row * N + col] = acc;
    }
}

// RoPE applied in-place to a [T x (H * 64)] buffer, per (row, head):
// q1 = q[:32], q2 = q[32:]; out = [q1*cos - q2*sin, q1*sin + q2*cos].
__global__ void rope_kernel(float* buf, const float* cos_t, const float* sin_t, int T,
                            int n_heads) {
    int row = blockIdx.x;
    int head = blockIdx.y;
    int i = threadIdx.x;  // 0..31
    if (i >= ROPE_HALF) return;

    float* q = buf + (long long)row * n_heads * D_HEAD + head * D_HEAD;
    float c = cos_t[(long long)row * ROPE_HALF + i];
    float s = sin_t[(long long)row * ROPE_HALF + i];
    float q1 = q[i];
    float q2 = q[i + ROPE_HALF];
    q[i] = q1 * c - q2 * s;
    q[i + ROPE_HALF] = q1 * s + q2 * c;
}

// Causal GQA attention per (row, q_head); K/V have N_KV_HEADS heads.
__global__ void gqa_attention_kernel(const float* Q, const float* K, const float* V, float* attn,
                                     int T) {
    __shared__ float q[D_HEAD];
    __shared__ float acc[D_HEAD];
    __shared__ float sdata[128];

    int row = blockIdx.x;
    int qh = blockIdx.y;
    int kvh = qh / (N_Q_HEADS / N_KV_HEADS);
    int tid = threadIdx.x;
    int limit = row + 1;  // causal

    for (int i = tid; i < D_HEAD; i += blockDim.x) {
        q[i] = Q[(long long)row * D_MODEL + qh * D_HEAD + i];
        acc[i] = 0.0f;
    }
    __syncthreads();

    float scale = (1.0f / sqrtf((float)D_HEAD));

    float local_max = -FLT_MAX;
    for (int j = tid; j < limit; j += blockDim.x) {
        const float* k = K + (long long)j * KV_DIM + kvh * D_HEAD;
        float s = 0.0f;
        for (int i = 0; i < D_HEAD; i++) s += q[i] * k[i];
        local_max = fmaxf(local_max, s * scale);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < limit; j += blockDim.x) {
        const float* k = K + (long long)j * KV_DIM + kvh * D_HEAD;
        const float* v = V + (long long)j * KV_DIM + kvh * D_HEAD;
        float s = 0.0f;
        for (int i = 0; i < D_HEAD; i++) s += q[i] * k[i];
        float e = expf(s * scale - row_max);
        local_sum += e;
        for (int i = 0; i < D_HEAD; i++) atomicAdd(&acc[i], e * v[i]);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int i = tid; i < D_HEAD; i += blockDim.x) {
        attn[(long long)row * D_MODEL + qh * D_HEAD + i] = acc[i] / row_sum;
    }
}

__global__ void add_kernel(const float* a, const float* b, float* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// x, output, weights, cos, sin are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, const float* cos,
                      const float* sin, int seq_len) {
    int T = seq_len;
    float *xn, *Q, *K, *V, *attn, *proj, *x1, *hn, *gate, *up, *hidden;
    hipMalloc(&xn, (size_t)T * D_MODEL * sizeof(float));
    hipMalloc(&Q, (size_t)T * D_MODEL * sizeof(float));
    hipMalloc(&K, (size_t)T * KV_DIM * sizeof(float));
    hipMalloc(&V, (size_t)T * KV_DIM * sizeof(float));
    hipMalloc(&attn, (size_t)T * D_MODEL * sizeof(float));
    hipMalloc(&proj, (size_t)T * D_MODEL * sizeof(float));
    hipMalloc(&x1, (size_t)T * D_MODEL * sizeof(float));
    hipMalloc(&hn, (size_t)T * D_MODEL * sizeof(float));
    hipMalloc(&gate, (size_t)T * FFN_DIM * sizeof(float));
    hipMalloc(&up, (size_t)T * FFN_DIM * sizeof(float));
    hipMalloc(&hidden, (size_t)T * FFN_DIM * sizeof(float));

    dim3 t2(TILE, TILE);
    int threads = 256;
    long long n_model = (long long)T * D_MODEL;
    int blocks_model = (int)((n_model + threads - 1) / threads);

    rmsnorm_kernel<<<T, 256>>>(x, xn, weights + OFF_W1, T, D_MODEL);

    dim3 g_q((D_MODEL + TILE - 1) / TILE, (T + TILE - 1) / TILE);
    dim3 g_kv((KV_DIM + TILE - 1) / TILE, (T + TILE - 1) / TILE);
    matmul_wt_kernel<<<g_q, t2>>>(xn, weights + OFF_WQ, Q, T, D_MODEL, D_MODEL);
    matmul_wt_kernel<<<g_kv, t2>>>(xn, weights + OFF_WK, K, T, D_MODEL, KV_DIM);
    matmul_wt_kernel<<<g_kv, t2>>>(xn, weights + OFF_WV, V, T, D_MODEL, KV_DIM);

    dim3 g_rope_q(T, N_Q_HEADS);
    dim3 g_rope_k(T, N_KV_HEADS);
    rope_kernel<<<g_rope_q, ROPE_HALF>>>(Q, cos, sin, T, N_Q_HEADS);
    rope_kernel<<<g_rope_k, ROPE_HALF>>>(K, cos, sin, T, N_KV_HEADS);

    dim3 g_attn(T, N_Q_HEADS);
    gqa_attention_kernel<<<g_attn, 128>>>(Q, K, V, attn, T);

    matmul_wt_kernel<<<g_q, t2>>>(attn, weights + OFF_WO, proj, T, D_MODEL, D_MODEL);
    add_kernel<<<blocks_model, threads>>>(x, proj, x1, n_model);

    rmsnorm_kernel<<<T, 256>>>(x1, hn, weights + OFF_W2, T, D_MODEL);

    dim3 g_ffn((FFN_DIM + TILE - 1) / TILE, (T + TILE - 1) / TILE);
    matmul_wt_kernel<<<g_ffn, t2>>>(hn, weights + OFF_WGATE, gate, T, D_MODEL, FFN_DIM);
    matmul_wt_kernel<<<g_ffn, t2>>>(hn, weights + OFF_WUP, up, T, D_MODEL, FFN_DIM);
    long long n_ffn = (long long)T * FFN_DIM;
    silu_mul_kernel<<<(int)((n_ffn + threads - 1) / threads), threads>>>(gate, up, hidden, n_ffn);
    matmul_wt_kernel<<<g_q, t2>>>(hidden, weights + OFF_WDOWN, proj, T, FFN_DIM, D_MODEL);
    add_kernel<<<blocks_model, threads>>>(x1, proj, output, n_model);
    hipDeviceSynchronize();

    hipFree(xn);
    hipFree(Q);
    hipFree(K);
    hipFree(V);
    hipFree(attn);
    hipFree(proj);
    hipFree(x1);
    hipFree(hn);
    hipFree(gate);
    hipFree(up);
    hipFree(hidden);
}
