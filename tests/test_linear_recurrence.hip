#include "common.h"

extern "C" void solve(const float* a, const float* x, float* h, int B, int L);

int main() {
    std::vector<float> a = {0.5f, 0.5f, 0.5f, 0.5f, 1, 1, 1, 1};
    std::vector<float> x = {1, 0, 0, 0, 1, 1, 1, 1};
    float* da = to_dev(a);
    float* dx = to_dev(x);
    float* dh = dev_alloc<float>(8);
    solve(da, dx, dh, 2, 4);
    return check_close(from_dev(dh, 8), {1, 0.5f, 0.25f, 0.125f, 1, 2, 3, 4});
}
