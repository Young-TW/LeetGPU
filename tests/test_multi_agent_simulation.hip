#include "common.h"

extern "C" void solve(const float* agents, float* agents_next, int N);

int main() {
    // agents (3,4) and (0,0) are exactly 5.0 apart -> not neighbors (r < 5.0)
    std::vector<float> agents = {0, 0, 1, 0, 3, 4, 0, -1};
    float* d_in = to_dev(agents);
    float* d_out = dev_alloc<float>(8);
    solve(d_in, d_out, 2);
    return check_close(from_dev(d_out, 8), {1, 0, 1, 0, 3, 3, 0, -1});
}
