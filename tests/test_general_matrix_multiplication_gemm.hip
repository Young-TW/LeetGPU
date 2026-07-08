#include "common.h"

extern "C" void solve(const __half* A, const __half* B, __half* C, int M, int N, int K,
                      float alpha, float beta);

static std::vector<__half> to_half(const std::vector<float>& v) {
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); i++) h[i] = __float2half(v[i]);
    return h;
}

int main() {
    // M=2, K=3, N=2; C = 1.0 * A@B + 0.0 * C_init
    auto A = to_half({1, 2, 3, 4, 5, 6});
    auto B = to_half({1, 2, 3, 4, 5, 6});
    auto C = to_half({1, 1, 1, 1});
    __half* dA = to_dev(A);
    __half* dB = to_dev(B);
    __half* dC = to_dev(C);
    solve(dA, dB, dC, 2, 2, 3, 1.0f, 0.0f);
    auto got_h = from_dev(dC, 4);
    std::vector<float> got(4);
    for (int i = 0; i < 4; i++) got[i] = __half2float(got_h[i]);
    return check_close(got, {22, 28, 49, 64}, 0.1f, 0.01f);
}
