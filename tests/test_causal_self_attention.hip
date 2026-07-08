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
    if (check_close(from_dev(dO, 8),
                    {1, 2, 3, 4, 3.4898374f, 4.4898374f, 5.4898374f, 6.4898374f}))
        return 1;

    // Spec's example 2 claims [[3,4],[5,6]], but row 1 (q=[1,1]) scores both keys
    // equally (1/sqrt(2)), so softmax must average V -> [4,5]. Example 1's exact
    // value (3.4898374) confirms the formula; the spec's second row is a typo.
    std::vector<float> Q2 = {0, 0, 1, 1}, K2 = {1, 0, 0, 1}, V2 = {3, 4, 5, 6};
    float* dQ2 = to_dev(Q2);
    float* dK2 = to_dev(K2);
    float* dV2 = to_dev(V2);
    float* dO2 = dev_alloc<float>(4);
    solve(dQ2, dK2, dV2, dO2, 2, 2);
    return check_close(from_dev(dO2, 4), {3, 4, 4, 5}, 0.01f, 0.01f);
}
