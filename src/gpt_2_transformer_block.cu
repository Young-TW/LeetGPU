#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// GPT-2 124M block: pre-norm, 12 heads x 64 dims, GELU(tanh approx) FFN, eps=1e-5.
// Weight offsets in the packed buffer (all row-major):
#define D_MODEL 768
#define N_HEADS 12
#define D_HEAD 64
#define FFN_DIM 3072
#define OFF_G1 0
#define OFF_B1 768
#define OFF_WQKV 1536
#define OFF_BQKV 1771008
#define OFF_WATTN 1773312
#define OFF_BATTN 2363136
#define OFF_G2 2363904
#define OFF_B2 2364672
#define OFF_WFC 2365440
#define OFF_BFC 4724736
#define OFF_WPROJ 4727808
#define OFF_BPROJ 7087104

#define TILE 16

// LayerNorm over the last dimension; one block per row.
__global__ void layernorm_kernel(const float* in, float* out, const float* gamma,
                                 const float* beta, int T, int d) {
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = in + (long long)row * d;

    float sum = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) sum += x[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / d;
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float v = x[i] - mean;
        var_sum += v * v;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = (1.0f / sqrtf(sdata[0] / d + 1e-5f));
    __syncthreads();

    for (int i = tid; i < d; i += blockDim.x) {
        out[(long long)row * d + i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// C[T x N] = A[T x K] @ W[K x N] + b[N]
__global__ void matmul_bias_kernel(const float* A, const float* W, const float* b, float* C,
                                   int T, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Ws[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int w_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < T && a_col < K) ? A[(long long)row * K + a_col] : 0.0f;
        Ws[threadIdx.y][threadIdx.x] =
            (w_row < K && col < N) ? W[(long long)w_row * N + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += As[threadIdx.y][i] * Ws[i][threadIdx.x];
        __syncthreads();
    }

    if (row < T && col < N) {
        C[(long long)row * N + col] = acc + b[col];
    }
}

// Per (row, head) attention over qkv buffer [T x 2304]; Q at +0, K at +768, V at +1536.
// No causal mask.
__global__ void attention_kernel(const float* qkv, float* attn, int T) {
    __shared__ float q[D_HEAD];
    __shared__ float acc[D_HEAD];
    __shared__ float sdata[128];

    int row = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int off = head * D_HEAD;

    for (int i = tid; i < D_HEAD; i += blockDim.x) {
        q[i] = qkv[(long long)row * 3 * D_MODEL + off + i];
        acc[i] = 0.0f;
    }
    __syncthreads();

    float scale = (1.0f / sqrtf((float)D_HEAD));

    float local_max = -FLT_MAX;
    for (int j = tid; j < T; j += blockDim.x) {
        const float* k = qkv + (long long)j * 3 * D_MODEL + D_MODEL + off;
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
    for (int j = tid; j < T; j += blockDim.x) {
        const float* k = qkv + (long long)j * 3 * D_MODEL + D_MODEL + off;
        const float* v = qkv + (long long)j * 3 * D_MODEL + 2 * D_MODEL + off;
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
        attn[(long long)row * D_MODEL + off + i] = acc[i] / row_sum;
    }
}

__global__ void add_kernel(const float* a, const float* b, float* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void gelu_kernel(float* x, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);  // sqrt(2/pi)
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

// x, output, weights are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {
    int T = seq_len;
    float *xn, *qkv, *attn, *proj, *x1, *hn, *fc;
    cudaMalloc(&xn, (size_t)T * D_MODEL * sizeof(float));
    cudaMalloc(&qkv, (size_t)T * 3 * D_MODEL * sizeof(float));
    cudaMalloc(&attn, (size_t)T * D_MODEL * sizeof(float));
    cudaMalloc(&proj, (size_t)T * D_MODEL * sizeof(float));
    cudaMalloc(&x1, (size_t)T * D_MODEL * sizeof(float));
    cudaMalloc(&hn, (size_t)T * D_MODEL * sizeof(float));
    cudaMalloc(&fc, (size_t)T * FFN_DIM * sizeof(float));

    dim3 t2(TILE, TILE);
    long long n_model = (long long)T * D_MODEL;
    int threads = 256;
    int blocks_model = (int)((n_model + threads - 1) / threads);

    // x_norm = LN1(x)
    layernorm_kernel<<<T, 256>>>(x, xn, weights + OFF_G1, weights + OFF_B1, T, D_MODEL);
    // QKV = x_norm @ W_qkv + b_qkv
    dim3 g_qkv((3 * D_MODEL + TILE - 1) / TILE, (T + TILE - 1) / TILE);
    matmul_bias_kernel<<<g_qkv, t2>>>(xn, weights + OFF_WQKV, weights + OFF_BQKV, qkv, T, D_MODEL,
                                      3 * D_MODEL);
    // multi-head attention
    dim3 g_attn(T, N_HEADS);
    attention_kernel<<<g_attn, 128>>>(qkv, attn, T);
    // P = A @ W_attn + b_attn
    dim3 g_proj((D_MODEL + TILE - 1) / TILE, (T + TILE - 1) / TILE);
    matmul_bias_kernel<<<g_proj, t2>>>(attn, weights + OFF_WATTN, weights + OFF_BATTN, proj, T,
                                       D_MODEL, D_MODEL);
    // x' = x + P
    add_kernel<<<blocks_model, threads>>>(x, proj, x1, n_model);
    // h_norm = LN2(x')
    layernorm_kernel<<<T, 256>>>(x1, hn, weights + OFF_G2, weights + OFF_B2, T, D_MODEL);
    // F = GELU(h_norm @ W_fc + b_fc) @ W_proj + b_proj
    dim3 g_fc((FFN_DIM + TILE - 1) / TILE, (T + TILE - 1) / TILE);
    matmul_bias_kernel<<<g_fc, t2>>>(hn, weights + OFF_WFC, weights + OFF_BFC, fc, T, D_MODEL,
                                     FFN_DIM);
    long long n_ffn = (long long)T * FFN_DIM;
    gelu_kernel<<<(int)((n_ffn + threads - 1) / threads), threads>>>(fc, n_ffn);
    matmul_bias_kernel<<<g_proj, t2>>>(fc, weights + OFF_WPROJ, weights + OFF_BPROJ, proj, T,
                                       FFN_DIM, D_MODEL);
    // output = x' + F
    add_kernel<<<blocks_model, threads>>>(x1, proj, output, n_model);
    cudaDeviceSynchronize();

    cudaFree(xn);
    cudaFree(qkv);
    cudaFree(attn);
    cudaFree(proj);
    cudaFree(x1);
    cudaFree(hn);
    cudaFree(fc);
}
