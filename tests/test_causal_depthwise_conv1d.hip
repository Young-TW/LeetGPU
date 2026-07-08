#include "common.h"

extern "C" void solve(const float* x, const float* weight, const float* bias, float* output, int B,
                      int L, int D, int K);

int main() {
    // B=1, L=4, D=2, K=3, channels-last
    std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w = {1, 0, -1, 1, 1, 1};  // (2,3)
    std::vector<float> b = {0, 0};
    float* dx = to_dev(x);
    float* dw = to_dev(w);
    float* db = to_dev(b);
    float* dO = dev_alloc<float>(8);
    solve(dx, dw, db, dO, 1, 4, 2, 3);
    return check_close(from_dev(dO, 8), {1, 2, 3, 6, 4, 12, 4, 18});
}
