#include "common.h"
#include <hip/hip_fp16.h>

extern "C" void solve(const __half* A, const __half* B, __half* result, int N);

static std::vector<__half> to_half(const std::vector<float>& v) {
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); i++) h[i] = __float2half(v[i]);
    return h;
}

int main() {
    auto A = to_half({1, 2, 3, 4});
    auto B = to_half({5, 6, 7, 8});
    __half* dA = to_dev(A);
    __half* dB = to_dev(B);
    __half* dR = dev_alloc<__half>(1);
    solve(dA, dB, dR, 4);
    float got = __half2float(from_dev(dR, 1)[0]);
    return check_close({got}, {70}, 0.5f, 0.01f);
}
