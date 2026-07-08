#include "common.h"

extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size);

int main() {
    std::vector<float> in = {1, 2, 3, 4, 5}, k = {1, 0, -1};
    float* d_in = to_dev(in);
    float* d_k = to_dev(k);
    float* d_out = dev_alloc<float>(3);
    solve(d_in, d_k, d_out, 5, 3);
    if (check_close(from_dev(d_out, 3), {-2, -2, -2})) return 1;

    std::vector<float> in2 = {2, 4, 6, 8}, k2 = {0.5f, 0.2f};
    float* d_in2 = to_dev(in2);
    float* d_k2 = to_dev(k2);
    float* d_out2 = dev_alloc<float>(3);
    solve(d_in2, d_k2, d_out2, 4, 2);
    return check_close(from_dev(d_out2, 3), {1.8f, 3.2f, 4.6f});
}
