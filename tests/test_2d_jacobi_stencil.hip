#include "common.h"

extern "C" void solve(const float* input, float* output, int rows, int cols);

int main() {
    // linear ramp: stencil average equals the center value, so output == input
    std::vector<float> in(16);
    for (int i = 0; i < 16; i++) in[i] = i + 1;
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(16);
    solve(d_in, d_out, 4, 4);
    return check_close(from_dev(d_out, 16), in);
}
