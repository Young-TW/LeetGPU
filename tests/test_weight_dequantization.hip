#include "common.h"

extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE);

int main() {
    std::vector<float> X = {10, 10, 5, 5, 10, 10, 5, 5, 2, 2, 8, 8, 2, 2, 8, 8};  // 4x4
    std::vector<float> S = {0.5f, 2.0f, 4.0f, 0.25f};                             // 2x2
    float* dX = to_dev(X);
    float* dS = to_dev(S);
    float* dY = dev_alloc<float>(16);
    solve(dX, dS, dY, 4, 4, 2);
    return check_close(from_dev(dY, 16),
                       {5, 5, 10, 10, 5, 5, 10, 10, 8, 8, 2, 2, 8, 8, 2, 2});
}
