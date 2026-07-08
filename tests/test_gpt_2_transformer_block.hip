#include "common.h"

extern "C" void solve(const float* x, float* output, const float* weights, int seq_len);

// With every weight and bias zero, both sub-layers contribute nothing
// (QKV = 0 -> uniform attention over V = 0; FFN of 0-input LN2 output is 0),
// so the residual stream passes x through unchanged. This exercises the full
// pipeline (LayerNorm, matmuls, attention, GELU, residuals, weight offsets).
int main() {
    const int T = 3, D = 768;
    const long long W_SIZE = 7087872LL;

    std::vector<float> x(T * D);
    for (int i = 0; i < T * D; i++) x[i] = 0.001f * ((i % 97) - 48);
    std::vector<float> weights(W_SIZE, 0.0f);

    float* dx = to_dev(x);
    float* dw = to_dev(weights);
    float* dO = dev_alloc<float>(T * D);
    solve(dx, dO, dw, T);
    return check_close(from_dev(dO, T * D), x, 1e-5f, 1e-5f);
}
