#include "common.h"
#include <limits>

extern "C" void solve(const float* dist, float* output, int N);

int main() {
    float inf = std::numeric_limits<float>::infinity();
    std::vector<float> dist = {0,   5,   inf, 10,  inf, 0,   3,   inf,
                               inf, inf, 0,   1,   inf, inf, inf, 0};
    float* d_dist = to_dev(dist);
    float* d_out = dev_alloc<float>(16);
    solve(d_dist, d_out, 4);
    std::vector<float> want = {0,   5,   8,   9,   inf, 0,   3,   4,
                               inf, inf, 0,   1,   inf, inf, inf, 0};
    return check_close(from_dev(d_out, 16), want);
}
