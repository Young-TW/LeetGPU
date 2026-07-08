#include "common.h"

extern "C" void solve(const float* A, const float* B, float* output, int N);

int main() {
    std::vector<float> A = {1, 2, 3}, B = {4, 5, 6};
    float* dA = to_dev(A);
    float* dB = to_dev(B);
    float* dO = dev_alloc<float>(6);
    solve(dA, dB, dO, 3);
    return check_close(from_dev(dO, 6), {1, 4, 2, 5, 3, 6});
}
