#include "common.h"

extern "C" void solve(const float* input, float* output, int N);

int main() {
    std::vector<float> in = {0.5f, 1.0f, -0.5f};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(3);
    solve(d_in, d_out, 3);
    return check_close(from_dev(d_out, 3), {0.3112295f, 0.731059f, -0.1887705f});
}
