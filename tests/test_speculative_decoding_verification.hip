#include "common.h"

extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V);

int main() {
    // B=1, T=3, V=4; expected output [1, 3, 0, 0] per the spec walkthrough
    std::vector<int> draft_tokens = {1, 2, 0};
    std::vector<float> draft_probs = {
        0.10f, 0.60f, 0.20f, 0.10f,   // p0
        0.10f, 0.20f, 0.50f, 0.20f,   // p1
        0.25f, 0.25f, 0.25f, 0.25f};  // p2 (never reached)
    std::vector<float> target_probs = {
        0.10f, 0.50f, 0.20f, 0.20f,   // q0
        0.30f, 0.20f, 0.20f, 0.30f,   // q1
        0.25f, 0.25f, 0.25f, 0.25f};  // q2 (never reached)
    std::vector<float> uniform = {0.50f, 0.70f, 0.30f, 0.90f};

    int* d_dt = to_dev(draft_tokens);
    float* d_dp = to_dev(draft_probs);
    float* d_tp = to_dev(target_probs);
    float* d_u = to_dev(uniform);
    int* d_out = dev_alloc<int>(4);
    solve(d_dt, d_dp, d_tp, d_u, d_out, 1, 3, 4);
    return check_exact(from_dev(d_out, 4), std::vector<int>{1, 3, 0, 0});
}
