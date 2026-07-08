#include "common.h"

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps);

int main() {
    std::vector<float> in = {1, 2, 3, 4, 5, 6};  // N=3, C=2
    std::vector<float> gamma = {1, 1}, beta = {0, 0};
    float* d_in = to_dev(in);
    float* d_g = to_dev(gamma);
    float* d_b = to_dev(beta);
    float* d_out = dev_alloc<float>(6);
    solve(d_in, d_g, d_b, d_out, 3, 2, 1e-5f);
    if (check_close(from_dev(d_out, 6), {-1.2247f, -1.2247f, 0, 0, 1.2247f, 1.2247f}, 1e-3f,
                    1e-3f))
        return 1;

    std::vector<float> in2 = {0, 1, 2, 3};  // N=2, C=2
    std::vector<float> g2 = {2, 0.5f}, b2 = {1, -1};
    float* d_in2 = to_dev(in2);
    float* d_g2 = to_dev(g2);
    float* d_b2 = to_dev(b2);
    float* d_out2 = dev_alloc<float>(4);
    solve(d_in2, d_g2, d_b2, d_out2, 2, 2, 1e-5f);
    return check_close(from_dev(d_out2, 4), {-1, -1.5f, 3, -0.5f}, 1e-3f, 1e-3f);
}
