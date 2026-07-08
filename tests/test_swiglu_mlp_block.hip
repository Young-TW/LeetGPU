#include "common.h"

extern "C" void solve(const float* x, const float* W_gate, const float* W_up, const float* W_down,
                      float* output, int M, int d_model, int d_ffn);

int main() {
    // M=2, d_model=2, d_ffn=4
    std::vector<float> x = {1, 0, 0, 1};
    std::vector<float> Wg = {1, 0, 0, 0, 0, 1, 0, 0};  // 2x4
    std::vector<float> Wd = {1, 0, 0, 1, 0, 0, 0, 0};  // 4x2
    float* dx = to_dev(x);
    float* dWg = to_dev(Wg);
    float* dWu = to_dev(Wg);  // W_up == W_gate in the example
    float* dWd = to_dev(Wd);
    float* dO = dev_alloc<float>(4);
    solve(dx, dWg, dWu, dWd, dO, 2, 2, 4);
    return check_close(from_dev(dO, 4), {0.7311f, 0, 0, 0.7311f}, 1e-3f, 1e-3f);
}
