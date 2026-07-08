#include "common.h"

extern "C" void solve(const float* A, float* B, int N);

int main() {
    std::vector<float> A = {1, 2, 3, 4};
    float* dA = to_dev(A);
    float* dB = dev_alloc<float>(4);
    solve(dA, dB, 2);
    return check_close(from_dev(dB, 4), {1, 2, 3, 4});
}
