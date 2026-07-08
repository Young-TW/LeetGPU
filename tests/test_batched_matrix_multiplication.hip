#include "common.h"

extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K);

int main() {
    // BATCH=2, M=2, K=3, N=2
    std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> B = {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1};
    float* dA = to_dev(A);
    float* dB = to_dev(B);
    float* dC = dev_alloc<float>(8);
    solve(dA, dB, dC, 2, 2, 2, 3);
    return check_close(from_dev(dC, 8), {22, 28, 49, 64, 92, 68, 128, 95});
}
