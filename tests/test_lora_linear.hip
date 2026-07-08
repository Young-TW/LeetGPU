#include "common.h"

extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      float* output, int batch, int d_in, int d_out, int rank, float lora_scale);

int main() {
    // batch=2, d_in=4, d_out=3, rank=2, scale=0.5
    std::vector<float> x = {1, 0, -1, 2, 0, 1, 1, -1};
    std::vector<float> W = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};  // 3x4
    std::vector<float> A = {1, 0, 0, 0, 0, 1, 0, 0};              // 2x4
    std::vector<float> B = {1, 0, 0, 1, 0, 0};                    // 3x2
    float* dx = to_dev(x);
    float* dW = to_dev(W);
    float* dA = to_dev(A);
    float* dB = to_dev(B);
    float* dO = dev_alloc<float>(6);
    solve(dx, dW, dA, dB, dO, 2, 4, 3, 2, 0.5f);
    return check_close(from_dev(dO, 6), {1.5f, 0, -1, 0, 1.5f, 1});
}
