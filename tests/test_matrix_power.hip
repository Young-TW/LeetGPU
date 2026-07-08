#include "common.h"

extern "C" void solve(const float* input, float* output, int N, int P);

int main() {
    std::vector<float> A = {1, 2, 3, 4};
    float* dA = to_dev(A);
    float* dO = dev_alloc<float>(4);
    solve(dA, dO, 2, 3);
    if (check_close(from_dev(dO, 4), {37, 54, 81, 118})) return 1;

    std::vector<float> B = {1, 0, 2, 0, 1, 0, 3, 0, 0};
    float* dB = to_dev(B);
    float* dO2 = dev_alloc<float>(9);
    solve(dB, dO2, 3, 2);
    return check_close(from_dev(dO2, 9), {7, 0, 2, 0, 1, 0, 3, 0, 6});
}
