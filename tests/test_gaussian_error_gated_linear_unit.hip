#include "common.h"

extern "C" void solve(const float* input, float* output, int N);

int main() {
    std::vector<float> in = {1, 1};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(1);
    solve(d_in, d_out, 2);
    if (check_close(from_dev(d_out, 1), {0.8413447f})) return 1;

    std::vector<float> in2 = {2, -1, 1, 0.5f};
    float* d_in2 = to_dev(in2);
    float* d_out2 = dev_alloc<float>(2);
    solve(d_in2, d_out2, 4);
    return check_close(from_dev(d_out2, 2), {1.6826895f, -0.3457312f});
}
