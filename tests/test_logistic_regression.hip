#include "common.h"

extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features);

int main() {
    std::vector<float> X = {2,  1,  1,  2,  3,  3,  1.5f, 2.5f,
                            -1, -2, -2, -1, -1.5f, -2.5f, -3, -3};  // 8x2
    std::vector<float> y = {1, 1, 1, 0, 0, 0, 1, 0};
    float* dX = to_dev(X);
    float* dy = to_dev(y);
    float* db = dev_alloc<float>(2);
    solve(dX, dy, db, 8, 2);
    return check_close(from_dev(db, 2), {2.26f, -1.29f}, 0.05f, 0.01f);
}
