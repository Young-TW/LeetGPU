#include "common.h"

extern "C" void solve(const float* A, int N, float* out);

int main() {
    std::vector<float> A = {1, -2, 3, 0, -1, 4};
    float* dA = to_dev(A);
    float* d_out = dev_alloc<float>(6);
    solve(dA, 6, d_out);
    return check_close(from_dev(d_out, 6), {1, 3, 4, 0, 0, 0});
}
