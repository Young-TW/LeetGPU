#include "common.h"

extern "C" void solve(const int* token_ids, const int* position_ids, const float* token_embeddings,
                      const float* position_embeddings, const float* gamma, const float* beta,
                      float* output, int B, int T, int V, int P, int D, float eps);

int main() {
    // B=1, T=2, V=3, P=2, D=4
    std::vector<int> token_ids = {2, 0};
    std::vector<int> position_ids = {0, 1};
    std::vector<float> tok_emb = {1, 2, 3, 4, 0, 1, 0, -1, 2, 0, -2, 0};
    std::vector<float> pos_emb = {0, 0, 0, 0, 1, 0, -1, 0};
    std::vector<float> gamma = {1, 1, 1, 1}, beta = {0, 0, 0, 0};
    int* d_tid = to_dev(token_ids);
    int* d_pid = to_dev(position_ids);
    float* d_te = to_dev(tok_emb);
    float* d_pe = to_dev(pos_emb);
    float* d_g = to_dev(gamma);
    float* d_b = to_dev(beta);
    float* d_out = dev_alloc<float>(8);
    solve(d_tid, d_pid, d_te, d_pe, d_g, d_b, d_out, 1, 2, 3, 2, 4, 1e-5f);
    return check_close(from_dev(d_out, 8),
                       {1.4142f, 0.0f, -1.4142f, 0.0f, -0.5773f, -0.5773f, -0.5773f, 1.7320f},
                       1e-3f, 1e-3f);
}
