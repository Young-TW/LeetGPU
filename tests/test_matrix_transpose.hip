#include "common.h"

extern "C" void solve(const float* input, float* output, int rows, int cols);

int main() {
    std::vector<float> in = {1, 2, 3, 4, 5, 6};  // 2x3
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(6);
    solve(d_in, d_out, 2, 3);
    return check_close(from_dev(d_out, 6), {1, 4, 2, 5, 3, 6});
}
