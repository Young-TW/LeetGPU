#include "common.h"

extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps);

int main() {
    std::vector<float> in = {1, 2, 3, 4};
    float* d_in = to_dev(in);
    float* d_out = dev_alloc<float>(4);
    solve(d_in, 1.0f, 0.0f, d_out, 4, 1e-5f);
    if (check_close(from_dev(d_out, 4), {0.36514813f, 0.73029625f, 1.0954444f, 1.4605925f}))
        return 1;

    std::vector<float> in2 = {1, 2, 3};
    float* d_in2 = to_dev(in2);
    float* d_out2 = dev_alloc<float>(3);
    solve(d_in2, 1.0f, 0.0f, d_out2, 3, 1e-5f);
    return check_close(from_dev(d_out2, 3), {0.46290955f, 0.9258191f, 1.3887286f});
}
