#include "common.h"

extern "C" void solve(const float* X, float* Y, int N);

int main() {
    std::vector<float> in = {0, 1, -1, 2};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(4);
    solve(d_in, d_out, 4);
    return check_close(from_dev(d_out, 4), {0.5f, 0.7311f, 0.2689f, 0.8808f}, 1e-3f, 1e-3f);
}
