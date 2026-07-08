#include "common.h"

extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N,
                      int C, int H, int W, int G, float eps);

int main() {
    // N=1, C=4, H=2, W=2, G=2
    std::vector<float> X = {1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 6, 6, 6, 6};
    std::vector<float> gamma = {1, 1, 1, 1}, beta = {0, 0, 0, 0};
    float* dX = to_dev(X);
    float* dg = to_dev(gamma);
    float* db = to_dev(beta);
    float* dY = dev_alloc<float>(16);
    solve(dX, dg, db, dY, 1, 4, 2, 2, 2, 1e-5f);
    std::vector<float> want = {-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1};
    if (check_close(from_dev(dY, 16), want, 1e-3f, 1e-3f)) return 1;

    // N=1, C=2, H=1, W=2, G=2 (instance norm)
    std::vector<float> X2 = {1, 3, 2, 6};
    std::vector<float> g2 = {2, 1}, b2 = {0, 0};
    float* dX2 = to_dev(X2);
    float* dg2 = to_dev(g2);
    float* db2 = to_dev(b2);
    float* dY2 = dev_alloc<float>(4);
    solve(dX2, dg2, db2, dY2, 1, 2, 1, 2, 2, 1e-5f);
    return check_close(from_dev(dY2, 4), {-2, 2, -1, 1}, 1e-3f, 1e-3f);
}
