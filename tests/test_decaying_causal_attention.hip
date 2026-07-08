#include "common.h"

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma);

int main() {
    std::vector<float> Q = {1, 1, 0, 0, 1, 1, 0, 0};
    std::vector<float> K = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<float> V = {4, 8, 12, 16, 4, 8, 12, 16};
    float* dQ = to_dev(Q);
    float* dK = to_dev(K);
    float* dV = to_dev(V);
    float* dO = dev_alloc<float>(8);
    solve(dQ, dK, dV, dO, 2, 4, 0.5f);
    return check_close(from_dev(dO, 8), {2, 4, 6, 8, 3, 6, 9, 12});
}
