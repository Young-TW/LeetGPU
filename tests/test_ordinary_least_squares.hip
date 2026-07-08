#include "common.h"

extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features);

int main() {
    std::vector<float> X = {-0.23f, -0.23f, 1.52f, 0.77f,  -0.47f, 1.58f, -0.14f, 0.65f,
                            0.5f,   -1.91f, -1.72f, 0.24f, -0.46f, -0.47f, 0.54f};  // 5x3
    std::vector<float> y = {83.01f, 93.4f, 47.33f, -62.22f, 13.06f};
    float* dX = to_dev(X);
    float* dy = to_dev(y);
    float* db = dev_alloc<float>(3);
    solve(dX, dy, db, 5, 3);
    return check_close(from_dev(db, 3), {13.97f, 29.12f, 61.05f}, 0.05f, 0.01f);
}
