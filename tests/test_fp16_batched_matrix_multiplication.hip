#include "common.h"

extern "C" void solve(const __half* A, const __half* B, __half* C, int BATCH, int M, int N, int K);

static std::vector<__half> to_half(const std::vector<float>& v) {
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); i++) h[i] = __float2half(v[i]);
    return h;
}

int main() {
    // BATCH=2, M=2, K=3, N=2
    auto A = to_half({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto B = to_half({1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1});
    __half* dA = to_dev(A);
    __half* dB = to_dev(B);
    __half* dC = dev_alloc<__half>(8);
    solve(dA, dB, dC, 2, 2, 2, 3);
    auto got_h = from_dev(dC, 8);
    std::vector<float> got(8);
    for (int i = 0; i < 8; i++) got[i] = __half2float(got_h[i]);
    return check_close(got, {22, 28, 49, 64, 92, 68, 128, 95}, 0.2f, 0.01f);
}
