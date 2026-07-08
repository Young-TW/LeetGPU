#include "common.h"

extern "C" void solve(const float* A, const float* B, float* C, int M, int N);

int main() {
    std::vector<float> A = {1, 3, 5, 7}, B = {2, 4, 6, 8};
    float* dA = to_dev(A);
    float* dB = to_dev(B);
    float* dC = dev_alloc<float>(8);
    solve(dA, dB, dC, 4, 4);
    if (check_close(from_dev(dC, 8), {1, 2, 3, 4, 5, 6, 7, 8})) return 1;

    std::vector<float> A2 = {-1, 1, 3}, B2 = {2};
    float* dA2 = to_dev(A2);
    float* dB2 = to_dev(B2);
    float* dC2 = dev_alloc<float>(4);
    solve(dA2, dB2, dC2, 3, 1);
    return check_close(from_dev(dC2, 4), {-1, 1, 2, 3});
}
