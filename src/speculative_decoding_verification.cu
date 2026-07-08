#include <cuda_runtime.h>

// One thread per sequence: walk draft positions left-to-right, accept/reject,
// resample from the adjusted distribution (or sample a bonus token if all accepted).
__global__ void verify_kernel(const int* draft_tokens, const float* draft_probs,
                              const float* target_probs, const float* uniform_samples,
                              int* output_tokens, int B, int T, int V) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int* out = output_tokens + (long long)b * (T + 1);
    for (int i = 0; i <= T; i++) out[i] = 0;

    int accepted = 0;
    bool rejected = false;

    for (int i = 0; i < T && !rejected; i++) {
        int t = draft_tokens[(long long)b * T + i];
        const float* p = draft_probs + ((long long)b * T + i) * V;
        const float* q = target_probs + ((long long)b * T + i) * V;
        float u = uniform_samples[(long long)b * (T + 1) + i];

        float alpha = q[t] / p[t];
        if (alpha > 1.0f) alpha = 1.0f;

        if (u < alpha) {
            out[accepted++] = t;
        } else {
            // reject: resample from adj(v) ∝ max(0, q(v) - p(v))
            rejected = true;
            float r = uniform_samples[(long long)b * (T + 1) + T];

            float total = 0.0f;
            for (int v = 0; v < V; v++) {
                float d = q[v] - p[v];
                if (d > 0.0f) total += d;
            }

            int chosen = V - 1;
            if (total > 0.0f) {
                float cum = 0.0f;
                for (int v = 0; v < V; v++) {
                    float d = q[v] - p[v];
                    if (d > 0.0f) cum += d / total;
                    if (cum >= r) {
                        chosen = v;
                        break;
                    }
                }
            } else {
                // uniform fallback: smallest k with (k+1)/V >= r
                chosen = (int)(r * V);
                if (chosen >= V) chosen = V - 1;
            }
            out[accepted++] = chosen;
        }
    }

    if (!rejected) {
        // all accepted: bonus token from q at the last position
        const float* q = target_probs + ((long long)b * T + (T - 1)) * V;
        float r = uniform_samples[(long long)b * (T + 1) + T];
        float cum = 0.0f;
        int chosen = V - 1;
        for (int v = 0; v < V; v++) {
            cum += q[v];
            if (cum >= r) {
                chosen = v;
                break;
            }
        }
        out[accepted++] = chosen;
    }
}

// draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) {
    int threadsPerBlock = 64;
    int blocksPerGrid = (B + threadsPerBlock - 1) / threadsPerBlock;

    verify_kernel<<<blocksPerGrid, threadsPerBlock>>>(draft_tokens, draft_probs, target_probs,
                                                      uniform_samples, output_tokens, B, T, V);
    cudaDeviceSynchronize();
}
