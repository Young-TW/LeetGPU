#include "common.h"

extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples);

int main() {
    std::vector<float> y = {0.0625f, 0.25f, 0.5625f, 1.0f, 1.5625f, 2.25f, 3.0625f, 4.0f};
    float* dy = to_dev(y);
    float* dr = dev_alloc<float>(1);
    solve(dy, dr, 0.0f, 2.0f, 8);
    return check_close(from_dev(dr, 1), {3.1875f});
}
