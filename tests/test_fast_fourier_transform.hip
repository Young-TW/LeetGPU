#include "common.h"

extern "C" void solve(const float* signal, float* spectrum, int N);

int main() {
    // impulse -> flat spectrum
    std::vector<float> sig = {1, 0, 0, 0, 0, 0, 0, 0};  // N=4 complex interleaved
    float* d_sig = to_dev(sig);
    float* d_spec = dev_alloc<float>(8);
    solve(d_sig, d_spec, 4);
    if (check_close(from_dev(d_spec, 8), {1, 0, 1, 0, 1, 0, 1, 0})) return 1;

    std::vector<float> sig2 = {1, 0, 1, 0};  // N=2: [1+0j, 1+0j]
    float* d_sig2 = to_dev(sig2);
    float* d_spec2 = dev_alloc<float>(4);
    solve(d_sig2, d_spec2, 2);
    return check_close(from_dev(d_spec2, 4), {2, 0, 0, 0});
}
