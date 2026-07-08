#include "common.h"

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d);

int main() {
    std::vector<float> Q = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<float> K = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<float> V = {1, 2, 3, 4, 5, 6, 7, 8};
    float* dQ = to_dev(Q);
    float* dK = to_dev(K);
    float* dV = to_dev(V);
    float* dO = dev_alloc<float>(8);
    solve(dQ, dK, dV, dO, 2, 4);
    if (check_close(from_dev(dO, 8), {2.8461537f, 3.8461537f, 4.8461537f, 5.8461537f, 3.1538463f,
                                      4.1538463f, 5.1538463f, 6.1538463f}))
        return 1;

    std::vector<float> Q2 = {0, 0, 1, 1}, K2 = {1, 0, 0, 1}, V2 = {3, 4, 5, 6};
    float* dQ2 = to_dev(Q2);
    float* dK2 = to_dev(K2);
    float* dV2 = to_dev(V2);
    float* dO2 = dev_alloc<float>(4);
    solve(dQ2, dK2, dV2, dO2, 2, 2);
    return check_close(from_dev(dO2, 4), {4, 5, 4, 5});
}
