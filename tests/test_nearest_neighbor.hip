#include "common.h"

extern "C" void solve(const float* points, int* indices, int N);

int main() {
    std::vector<float> pts = {0, 0, 0, 1, 0, 0, 5, 5, 5};
    float* d_pts = to_dev(pts);
    int* d_idx = dev_alloc<int>(3);
    solve(d_pts, d_idx, 3);
    return check_exact(from_dev(d_idx, 3), std::vector<int>{1, 0, 1});
}
