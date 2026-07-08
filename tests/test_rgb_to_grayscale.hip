#include "common.h"

extern "C" void solve(const float* input, float* output, int width, int height);

int main() {
    std::vector<float> in = {255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(4);
    solve(d_in, d_out, 2, 2);
    if (check_close(from_dev(d_out, 4), {76.245f, 149.685f, 29.07f, 128.0f})) return 1;

    std::vector<float> in2 = {100, 150, 200};
    float* d_in2 = to_dev(in2);
    float* d_out2 = dev_alloc<float>(1);
    solve(d_in2, d_out2, 1, 1);
    return check_close(from_dev(d_out2, 1), {140.75f});
}
