#include "common.h"

extern "C" void solve(const float* input, float* output, float lo, float hi, int N);

int main() {
    std::vector<float> in = {1.5f, -2.0f, 3.0f, 4.5f};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(4);
    solve(d_in, d_out, 0.0f, 3.5f, 4);
    if (check_close(from_dev(d_out, 4), {1.5f, 0.0f, 3.0f, 3.5f})) return 1;

    std::vector<float> in2 = {-1, 2, 5};
    float* d_in2 = to_dev(in2);
    float* d_out2 = dev_alloc<float>(3);
    solve(d_in2, d_out2, -0.5f, 2.5f, 3);
    return check_close(from_dev(d_out2, 3), {-0.5f, 2.0f, 2.5f});
}
