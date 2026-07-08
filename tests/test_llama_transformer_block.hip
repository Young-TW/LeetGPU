#include "common.h"

extern "C" void solve(const float* x, float* output, const float* weights, const float* cos,
                      const float* sin, int seq_len);

// All-zero weights make both sub-layers contribute nothing (RMSNorm scale 0 zeroes
// the normed input, so Q=K=V=0 and the FFN output is 0); output must equal x.
int main() {
    const int T = 3, D = 512, RH = 32;
    const long long W_SIZE = 2819072LL;

    std::vector<float> x(T * D);
    for (int i = 0; i < T * D; i++) x[i] = 0.001f * ((i % 89) - 44);
    std::vector<float> weights(W_SIZE, 0.0f);
    std::vector<float> cos_t(T * RH, 1.0f), sin_t(T * RH, 0.0f);

    float* dx = to_dev(x);
    float* dw = to_dev(weights);
    float* dc = to_dev(cos_t);
    float* ds = to_dev(sin_t);
    float* dO = dev_alloc<float>(T * D);
    solve(dx, dO, dw, dc, ds, T);
    return check_close(from_dev(dO, T * D), x, 1e-5f, 1e-5f);
}
