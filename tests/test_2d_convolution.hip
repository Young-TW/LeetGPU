#include "common.h"

extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols);

int main() {
    std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // 3x3
    std::vector<float> k = {0, 1, 1, 0};                  // 2x2
    float* d_in = to_dev(in);
    float* d_k = to_dev(k);
    float* d_out = dev_alloc<float>(4);
    solve(d_in, d_k, d_out, 3, 3, 2, 2);
    if (check_close(from_dev(d_out, 4), {6, 8, 12, 14})) return 1;

    // 4x4 input, 1x3 kernel -> 4x2
    std::vector<float> in2 = {1, 1, 1, 1, 1, 2, 3, 1, 1, 4, 5, 1, 1, 1, 1, 1};
    std::vector<float> k2 = {1, 0, 1};
    float* d_in2 = to_dev(in2);
    float* d_k2 = to_dev(k2);
    float* d_out2 = dev_alloc<float>(8);
    solve(d_in2, d_k2, d_out2, 4, 4, 1, 3);
    return check_close(from_dev(d_out2, 8), {2, 2, 4, 3, 6, 5, 2, 2});
}
