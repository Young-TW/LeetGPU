#include "common.h"

extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols);

int main() {
    // 2x2x2 volume [1..8], 2x2x2 kernel of ones -> [36]
    std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> k(8, 1.0f);
    float* d_in = to_dev(in);
    float* d_k = to_dev(k);
    float* d_out = dev_alloc<float>(1);
    solve(d_in, d_k, d_out, 2, 2, 2, 2, 2, 2);
    return check_close(from_dev(d_out, 1), {36});
}
