#include "common.h"

extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D);

int main() {
    std::vector<float> Q = {1, 2, 3, 4, 1, 1, 1, 1};
    std::vector<float> c = {1, 1, 1, 1, 0, 0, 0, 0};
    std::vector<float> s = {0, 0, 0, 0, 1, 1, 1, 1};
    float* dQ = to_dev(Q);
    float* dc = to_dev(c);
    float* ds = to_dev(s);
    float* dO = dev_alloc<float>(8);
    solve(dQ, dc, ds, dO, 2, 4);
    return check_close(from_dev(dO, 8), {1, 2, 3, 4, -1, -1, 1, 1});
}
