#include "common.h"

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h);

int main() {
    // N=2, d_model=4, h=2
    std::vector<float> Q = {1, 0, 2, 3, 4, 5, 6, 7};
    std::vector<float> K = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> V = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};
    float* dQ = to_dev(Q);
    float* dK = to_dev(K);
    float* dV = to_dev(V);
    float* dO = dev_alloc<float>(8);
    solve(dQ, dK, dV, dO, 2, 4, 2);
    if (check_close(from_dev(dO, 8), {2.39f, 2.89f, 3.50f, 4.00f, 2.50f, 3.00f, 3.50f, 4.00f},
                    0.01f, 0.01f))
        return 1;

    // N=1, d_model=2, h=1
    std::vector<float> Q2 = {1, 1}, K2 = {1, 1}, V2 = {2, 3};
    float* dQ2 = to_dev(Q2);
    float* dK2 = to_dev(K2);
    float* dV2 = to_dev(V2);
    float* dO2 = dev_alloc<float>(2);
    solve(dQ2, dK2, dV2, dO2, 1, 2, 1);
    return check_close(from_dev(dO2, 2), {2, 3});
}
