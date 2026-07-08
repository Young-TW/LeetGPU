#include "common.h"

extern "C" void solve(const float* u, const float* delta, const float* A, const float* B,
                      const float* C, const float* skip, float* y, int batch, int seq_len,
                      int d_model, int d_state);

int main() {
    // batch=1, seq_len=4, d_model=2, d_state=2
    std::vector<float> u = {1, 0, 0, 1, 1, 1, 0, 0};
    std::vector<float> delta(8, 1.0f);
    std::vector<float> A = {-0.5f, -1.0f, -0.5f, -1.0f};
    std::vector<float> B = {1, 0, 0, 1, 1, 1, 0.5f, 0.5f};
    std::vector<float> C = {1, 0, 0, 1, 1, 1, 0.5f, 0.5f};
    std::vector<float> skip = {0, 0};
    float* du = to_dev(u);
    float* dd = to_dev(delta);
    float* dA = to_dev(A);
    float* dB = to_dev(B);
    float* dC = to_dev(C);
    float* ds = to_dev(skip);
    float* dy = dev_alloc<float>(8);
    solve(du, dd, dA, dB, dC, ds, dy, 1, 4, 2, 2);
    return check_close(from_dev(dy, 8), {1.000f, 0.000f, 0.000f, 1.000f, 2.368f, 2.368f, 0.599f,
                                         0.555f},
                       1e-3f, 1e-3f);
}
