#include "common.h"

extern "C" void solve(const float* predictions, const float* targets, float* mse, int N);

int main() {
    std::vector<float> p = {1, 2, 3, 4}, t = {1.5f, 2.5f, 3.5f, 4.5f};
    float* dp = to_dev(p);
    float* dt = to_dev(t);
    float* dm = dev_alloc<float>(1);
    solve(dp, dt, dm, 4);
    if (check_close(from_dev(dm, 1), {0.25f})) return 1;

    std::vector<float> p2 = {10, 20, 30}, t2 = {12, 18, 33};
    float* dp2 = to_dev(p2);
    float* dt2 = to_dev(t2);
    float* dm2 = dev_alloc<float>(1);
    solve(dp2, dt2, dm2, 3);
    return check_close(from_dev(dm2, 1), {5.6667f}, 1e-3f, 1e-3f);
}
