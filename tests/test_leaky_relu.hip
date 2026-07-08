#include "common.h"

extern "C" void solve(const float* input, float* output, int N);

int main() {
    std::vector<float> in = {1, -2, 3, -4};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(4);
    solve(d_in, d_out, 4);
    return check_close(from_dev(d_out, 4), {1, -0.02f, 3, -0.04f});
}
