#include "common.h"

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha);

int main() {
    // Example 1 (the spec's first output row reads 3.05,4.05,6.05,7.05 which is
    // impossible for a convex combination of the V rows; the corrected row is
    // 3.05,4.05,5.05,6.05 by direct computation).
    std::vector<float> Q = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<float> K = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::vector<float> V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float* dQ = to_dev(Q);
    float* dK = to_dev(K);
    float* dV = to_dev(V);
    float* dO = dev_alloc<float>(8);
    solve(dQ, dK, dV, dO, 2, 3, 4, 0.5f);
    if (check_close(from_dev(dO, 8), {3.05f, 4.05f, 5.05f, 6.05f, 3.93f, 4.93f, 5.93f, 6.93f},
                    0.01f, 0.01f))
        return 1;

    std::vector<float> Q2 = {1, 2}, K2 = {1, 0, 0, 1}, V2 = {3, 4, 5, 6};
    float* dQ2 = to_dev(Q2);
    float* dK2 = to_dev(K2);
    float* dV2 = to_dev(V2);
    float* dO2 = dev_alloc<float>(2);
    solve(dQ2, dK2, dV2, dO2, 1, 2, 2, 0.8f);
    return check_close(from_dev(dO2, 2), {3.95f, 4.95f}, 0.01f, 0.01f);
}
