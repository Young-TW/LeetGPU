#include "common.h"

extern "C" void solve(const float* input, float* output, int N);

int main() {
    std::vector<float> in = {1, 2, 3, 4};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(4);
    solve(d_in, d_out, 4);
    if (check_close(from_dev(d_out, 4), {1, 3, 6, 10})) return 1;

    std::vector<float> in2 = {5, -2, 3, 1, -4};
    float* d_in2 = to_dev(in2);
    float* d_out2 = dev_alloc<float>(5);
    solve(d_in2, d_out2, 5);
    if (check_close(from_dev(d_out2, 5), {5, 3, 6, 7, 3})) return 1;

    // larger case spanning multiple scan blocks: 5000 ones
    std::vector<float> in3(5000, 1.0f);
    std::vector<float> want3(5000);
    for (int i = 0; i < 5000; i++) want3[i] = i + 1;
    float* d_in3 = to_dev(in3);
    float* d_out3 = dev_alloc<float>(5000);
    solve(d_in3, d_out3, 5000);
    return check_close(from_dev(d_out3, 5000), want3);
}
