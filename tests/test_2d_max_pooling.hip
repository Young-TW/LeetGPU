#include "common.h"

extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding);

int main() {
    // N=1, C=1, 3x3, k=2, s=1, p=0 -> 2x2
    std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(4);
    solve(d_in, d_out, 1, 1, 3, 3, 2, 1, 0);
    if (check_close(from_dev(d_out, 4), {5, 6, 8, 9})) return 1;

    // N=1, C=1, 5x5, k=3, s=1, p=1 -> 5x5
    std::vector<float> in2(25);
    for (int i = 0; i < 25; i++) in2[i] = i + 1;
    std::vector<float> want2 = {7,  8,  9,  10, 10, 12, 13, 14, 15, 15, 17, 18, 19,
                                20, 20, 22, 23, 24, 25, 25, 22, 23, 24, 25, 25};
    float* d_in2 = to_dev(in2);
    float* d_out2 = dev_alloc<float>(25);
    solve(d_in2, d_out2, 1, 1, 5, 5, 3, 1, 1);
    return check_close(from_dev(d_out2, 25), want2);
}
