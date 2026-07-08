#include "common.h"

extern "C" void solve(const float* signal, float* spectrum, int M, int N);

int main() {
    // 2x2 impulse -> all-ones spectrum
    std::vector<float> sig = {1, 0, 0, 0, 0, 0, 0, 0};
    float* d_sig = to_dev(sig);
    float* d_spec = dev_alloc<float>(8);
    solve(d_sig, d_spec, 2, 2);
    return check_close(from_dev(d_spec, 8), {1, 0, 1, 0, 1, 0, 1, 0});
}
