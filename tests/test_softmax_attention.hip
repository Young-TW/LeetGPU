#include "common.h"

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d);

int main() {
    // M=2, N=3, d=4
    std::vector<float> Q = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<float> K = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::vector<float> V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float* dQ = to_dev(Q);
    float* dK = to_dev(K);
    float* dV = to_dev(V);
    float* dO = dev_alloc<float>(8);
    solve(dQ, dK, dV, dO, 2, 3, 4);
    return check_close(from_dev(dO, 8), {4.29f, 5.29f, 6.29f, 7.29f, 5.00f, 6.00f, 7.00f, 8.00f},
                       0.01f, 0.01f);
}
