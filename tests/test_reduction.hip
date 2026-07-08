#include "common.h"

extern "C" void solve(const float* input, float* output, int N);

int main() {
    std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(1);
    solve(d_in, d_out, 8);
    if (check_close(from_dev(d_out, 1), {36})) return 1;

    std::vector<float> in2 = {-2.5f, 1.5f, -1.0f, 2.0f};
    float* d_in2 = to_dev(in2);
    float* d_out2 = dev_alloc<float>(1);
    solve(d_in2, d_out2, 4);
    return check_close(from_dev(d_out2, 1), {0});
}
