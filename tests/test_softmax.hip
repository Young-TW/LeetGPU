#include "common.h"

extern "C" void solve(const float* input, float* output, int N);

int main() {
    std::vector<float> in = {1, 2, 3};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(3);
    solve(d_in, d_out, 3);
    return check_close(from_dev(d_out, 3), {0.09003057f, 0.24472847f, 0.66524096f});
}
