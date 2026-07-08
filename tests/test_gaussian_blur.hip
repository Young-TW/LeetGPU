#include "common.h"

extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols);

int main() {
    std::vector<float> img = {10, 20, 30, 40, 50, 60, 70, 80, 90};  // 3x3
    std::vector<float> k = {0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.1f};
    float* d_img = to_dev(img);
    float* d_k = to_dev(k);
    float* d_out = dev_alloc<float>(9);
    solve(d_img, d_k, d_out, 3, 3, 3, 3);
    return check_close(from_dev(d_out, 9), {13, 23, 19, 31, 50, 39, 31, 47, 37});
}
