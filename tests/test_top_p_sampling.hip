#include "common.h"

extern "C" void solve(const float* logits, const float* p, const int* seed, int* sampled_token,
                      int vocab_size);

int main() {
    // Example 1: nucleus for p=0.9 over softmax([1,2,3,0.5]) is {2,1,0}
    // (cumulative 0.631, 0.863, 0.948) — accept any of those.
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 0.5f};
    std::vector<float> p = {0.9f};
    std::vector<int> seed = {42};
    float* d_logits = to_dev(logits);
    float* d_p = to_dev(p);
    int* d_seed = to_dev(seed);
    int* d_tok = dev_alloc<int>(1);
    solve(d_logits, d_p, d_seed, d_tok, 4);
    int tok = from_dev(d_tok, 1)[0];
    if (tok != 0 && tok != 1 && tok != 2) {
        printf("FAIL: token %d outside nucleus {0,1,2}\n", tok);
        return 1;
    }

    // Example 2: token 0 dominates; p=0.5 -> must sample 0
    std::vector<float> logits2 = {10.0f, 1.0f, 1.0f};
    std::vector<float> p2 = {0.5f};
    std::vector<int> seed2 = {123};
    float* d_logits2 = to_dev(logits2);
    float* d_p2 = to_dev(p2);
    int* d_seed2 = to_dev(seed2);
    int* d_tok2 = dev_alloc<int>(1);
    solve(d_logits2, d_p2, d_seed2, d_tok2, 3);
    int tok2 = from_dev(d_tok2, 1)[0];
    if (tok2 != 0) {
        printf("FAIL: expected token 0, got %d\n", tok2);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
