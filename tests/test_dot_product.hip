#include "common.h"

extern "C" void solve(const float* A, const float* B, float* result, int N);

int main() {
    std::vector<float> A = {1, 2, 3, 4}, B = {5, 6, 7, 8};
    float* dA = to_dev(A);
    float* dB = to_dev(B);
    float* dR = dev_alloc<float>(1);
    solve(dA, dB, dR, 4);
    return check_close(from_dev(dR, 1), {70});
}
