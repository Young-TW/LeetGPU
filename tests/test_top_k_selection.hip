#include "common.h"

extern "C" void solve(const float* input, float* output, int N, int k);

int main() {
    std::vector<float> in = {1, 5, 3, 2, 4};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(3);
    solve(d_in, d_out, 5, 3);
    if (check_close(from_dev(d_out, 3), {5, 4, 3})) return 1;

    std::vector<float> in2 = {7.2f, -1.0f, 3.3f, 8.8f, 2.2f};
    float* d_in2 = to_dev(in2);
    float* d_out2 = dev_alloc<float>(2);
    solve(d_in2, d_out2, 5, 2);
    return check_close(from_dev(d_out2, 2), {8.8f, 7.2f});
}
