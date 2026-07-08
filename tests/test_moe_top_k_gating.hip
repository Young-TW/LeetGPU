#include "common.h"

extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k);

int main() {
    std::vector<float> logits = {1, 2, 3, 4, 4, 3, 2, 1};  // 2x4
    float* d_logits = to_dev(logits);
    float* d_w = dev_alloc<float>(4);
    int* d_i = dev_alloc<int>(4);
    solve(d_logits, d_w, d_i, 2, 4, 2);
    if (check_exact(from_dev(d_i, 4), std::vector<int>{3, 2, 0, 1})) return 1;
    return check_close(from_dev(d_w, 4), {0.7311f, 0.2689f, 0.7311f, 0.2689f}, 1e-3f, 1e-3f);
}
