#include "common.h"

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz);

int main() {
    std::vector<float> A = {2, 0, 0, 1, 0, 3, 0, 0, 0, 0, 4, 0};  // 3x4
    std::vector<float> B = {1, 2, 3, 4, 5, 6, 7, 8};              // 4x2
    float* dA = to_dev(A);
    float* dB = to_dev(B);
    float* dC = dev_alloc<float>(6);
    solve(dA, dB, dC, 3, 4, 2, 4);
    return check_close(from_dev(dC, 6), {9, 12, 9, 12, 20, 24});
}
