#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <math.h>

// 10-parameter adder transformer (AdderBoard). One thread per batch sequence:
// 11 autoregressive decode steps, each recomputing the single-layer forward pass
// for the last position only (K/V depend only on the embeddings, not on other
// positions' attention outputs).

#define PROMPT_LEN 31
#define DECODE_STEPS 11
#define MAX_SEQ (PROMPT_LEN + DECODE_STEPS)
#define EPS 1e-6f

__device__ inline void unit_rmsnorm(float& x0, float& x1) {
    float inv = (1.0f / sqrtf(0.5f * (x0 * x0 + x1 * x1) + EPS));
    x0 *= inv;
    x1 *= inv;
}

__global__ void adder_kernel(const int* prompts, float* output, const float* weights,
                             int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    float w0 = weights[0], w1 = weights[1];
    float q0 = weights[2], q1 = weights[3];
    float v0 = weights[4];
    float ga = weights[5], gc = weights[6];
    float carry = weights[7];
    float n0 = weights[8], n1 = weights[9];

    const float omega = 2.0f * (float)M_PI / 19.0f;
    // S^2 = ln(10) / (sqrt(2) * (cos(0.3w) - cos(0.7w))); scale = S^2 / sqrt(2)
    float S2 = logf(10.0f) / (sqrtf(2.0f) * (cosf(0.3f * omega) - cosf(0.7f * omega)));
    float scale = S2 / sqrtf(2.0f);

    int tokens[MAX_SEQ];
    for (int i = 0; i < PROMPT_LEN; i++) tokens[i] = prompts[(long long)b * PROMPT_LEN + i];

    for (int step = 0; step < DECODE_STEPS; step++) {
        int seq_len = PROMPT_LEN + step;
        int last = seq_len - 1;

        // last-position Q (normed embedding -> projection -> unit norm -> RoPE)
        float el0, el1;
        {
            int d = tokens[last];
            el0 = w0 - w1 * d * d;
            el1 = -(float)d;
        }
        float h0 = el0, h1 = el1;
        unit_rmsnorm(h0, h1);

        float Qx = h0 * q0, Qy = h0 * q1;
        unit_rmsnorm(Qx, Qy);
        {
            float pw = last * omega;
            float c = cosf(pw), s = sinf(pw);
            float r0 = Qx * c - Qy * s;
            float r1 = Qx * s + Qy * c;
            Qx = r0;
            Qy = r1;
        }

        // causal attention over all positions: online softmax over j = 0..last
        float m = -1e30f, l = 0.0f, acc0 = 0.0f;  // V has only component 0 non-zero
        for (int j = 0; j <= last; j++) {
            int d = tokens[j];
            float e0 = w0 - w1 * d * d;
            float e1 = -(float)d;
            float g0 = e0, g1 = e1;
            unit_rmsnorm(g0, g1);

            float Kx = g0, Ky = 0.0f;
            unit_rmsnorm(Kx, Ky);
            float pw = j * omega;
            float c = cosf(pw), s = sinf(pw);
            float k0 = Kx * c - Ky * s;
            float k1 = Kx * s + Ky * c;

            float Vv = g1 * v0;

            float score = scale * (Qx * k0 + Qy * k1);
            if (score > m) {
                float f = expf(m - score);
                l = l * f;
                acc0 = acc0 * f;
                m = score;
            }
            float e = expf(score - m);
            l += e;
            acc0 += e * Vv;
        }
        float attn0 = acc0 / l;

        // output projection [attn0, attn1] -> [0, attn0]; residual on the embedding
        float x0 = el0, x1 = el1 + attn0;

        // MLP on unit-normed hidden state, plus residual
        float hh0 = x0, hh1 = x1;
        unit_rmsnorm(hh0, hh1);
        float g0 = hh0 * ga + hh1 * gc;
        float g1 = hh0 * (ga - gc / 1000.0f) + hh1 * gc;
        float base = hh0;
        float mix0 = (g0 / (1.0f + expf(-g0))) * base;
        float mix1 = (g1 / (1.0f + expf(-g1))) * base;
        x1 += carry * (mix1 - mix0);

        // final RMSNorm with learned weight
        float inv = (1.0f / sqrtf(0.5f * (x0 * x0 + x1 * x1) + EPS));
        float o0 = x0 * inv * n0;
        float o1 = x1 * inv * n1;

        // logits = out . e(d) over the tied embedding, track argmax
        float best = -1e30f;
        int best_d = 0;
        for (int d = 0; d < 10; d++) {
            float logit = o0 * (w0 - w1 * d * d) + o1 * (-(float)d);
            output[((long long)b * DECODE_STEPS + step) * 10 + d] = logit;
            if (logit > best) {
                best = logit;
                best_d = d;
            }
        }
        tokens[seq_len] = best_d;
    }
}

// prompts, output, weights are device pointers
extern "C" void solve(const int* prompts, float* output, const float* weights, int batch_size) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    adder_kernel<<<blocksPerGrid, threadsPerBlock>>>(prompts, output, weights, batch_size);
    hipDeviceSynchronize();
}
