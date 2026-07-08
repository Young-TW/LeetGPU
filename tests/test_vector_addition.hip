#include "common.h"

extern "C" void solve(const float* A, const float* B, float* C, int N);

int main() {
    std::vector<float> A = {1, 2, 3, 4}, B = {5, 6, 7, 8};
    float* dA = to_dev(A);
    float* dB = to_dev(B);
    float* dC = dev_alloc<float>(4);
    solve(dA, dB, dC, 4);
    return check_close(from_dev(dC, 4), {6, 8, 10, 12});
}
