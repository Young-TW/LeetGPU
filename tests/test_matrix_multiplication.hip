#include "common.h"

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K);

int main() {
    // A (2x2) * B (2x2): M=2, N=2, K=2
    std::vector<float> A = {1, 2, 3, 4}, B = {5, 6, 7, 8};
    float* dA = to_dev(A);
    float* dB = to_dev(B);
    float* dC = dev_alloc<float>(4);
    solve(dA, dB, dC, 2, 2, 2);
    if (check_close(from_dev(dC, 4), {19, 22, 43, 50})) return 1;

    // A (1x3) * B (3x1)
    std::vector<float> A2 = {1, 2, 3}, B2 = {4, 5, 6};
    float* dA2 = to_dev(A2);
    float* dB2 = to_dev(B2);
    float* dC2 = dev_alloc<float>(1);
    solve(dA2, dB2, dC2, 1, 3, 1);
    return check_close(from_dev(dC2, 1), {32});
}
