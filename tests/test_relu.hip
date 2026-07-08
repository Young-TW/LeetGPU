#include "common.h"

extern "C" void solve(const float* input, float* output, int N);

int main() {
    std::vector<float> in = {-2, -1, 0, 1, 2};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(5);
    solve(d_in, d_out, 5);
    return check_close(from_dev(d_out, 5), {0, 0, 0, 1, 2});
}
