#include "common.h"
#include <stdint.h>

extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size);

static std::vector<__half> to_half(const std::vector<float>& v) {
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); i++) h[i] = __float2half(v[i]);
    return h;
}

int main() {
    // M=2, N=4, K=4, group_size=2
    auto x = to_half({1, 0, 1, 0, 0, 1, 0, 1});
    std::vector<uint8_t> w_q = {0x99, 0x99, 0xAA, 0xAA, 0x77, 0x77, 0x88, 0x88};
    auto scales = to_half(std::vector<float>(8, 0.5f));
    __half* dx = to_dev(x);
    uint8_t* dw = to_dev(w_q);
    __half* ds = to_dev(scales);
    __half* dy = dev_alloc<__half>(8);
    solve(dx, dw, ds, dy, 2, 4, 4, 2);
    auto got_h = from_dev(dy, 8);
    std::vector<float> got(8);
    for (int i = 0; i < 8; i++) got[i] = __half2float(got_h[i]);
    return check_close(got, {1, 2, -1, 0, 1, 2, -1, 0}, 0.05f, 0.01f);
}
