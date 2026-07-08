#include "common.h"

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int window_size);

int main() {
    std::vector<float> Q = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<float> K = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<float> V = {1, 2, 3, 4, 5, 6, 7, 8};
    float* dQ = to_dev(Q);
    float* dK = to_dev(K);
    float* dV = to_dev(V);
    float* dO = dev_alloc<float>(8);
    solve(dQ, dK, dV, dO, 2, 4, 1);
    if (check_close(from_dev(dO, 8), {2.5101628f, 3.5101628f, 4.510163f, 5.510163f, 3.4898374f,
                                      4.4898376f, 5.4898376f, 6.489837f}))
        return 1;

    std::vector<float> Q2 = {0, 0, 0, 0, 1, 0};
    std::vector<float> K2 = {1, 0, 0, 0, 1, 0};
    std::vector<float> V2 = {1, 2, 3, 5, 6, 7};
    float* dQ2 = to_dev(Q2);
    float* dK2 = to_dev(K2);
    float* dV2 = to_dev(V2);
    float* dO2 = dev_alloc<float>(6);
    solve(dQ2, dK2, dV2, dO2, 2, 3, 1);
    return check_close(from_dev(dO2, 6),
                       {3, 4, 5, 3.5618298f, 4.56183f, 5.5618296f});
}
