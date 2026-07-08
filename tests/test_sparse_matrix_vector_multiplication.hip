#include "common.h"

extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz);

int main() {
    std::vector<float> A = {5, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 4};  // 3x4
    std::vector<float> x = {1, 2, 3, 4};
    float* dA = to_dev(A);
    float* dx = to_dev(x);
    float* dy = dev_alloc<float>(3);
    solve(dA, dx, dy, 3, 4, 6);
    return check_close(from_dev(dy, 3), {9, 13, 16});
}
