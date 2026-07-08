#include "common.h"

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim);

int main() {
    // num_q_heads=4, num_kv_heads=2, seq_len=3, head_dim=4; layout [head, seq, dim]
    std::vector<float> Q = {
        1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,                          // Q0
        0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,                          // Q1
        -1, 0, 0.5f, 0, 0, -1, 0, 0.5f, 0.5f, 0, -1, 0,              // Q2
        0, 0.5f, 0, -1, 0.5f, 0, 0, -1, 0, 0, 0.5f, 0.5f};           // Q3
    std::vector<float> K = {
        1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,                          // K0
        0, 1, 0, -1, -1, 0, 1, 0, 0, -1, 0, 1};                      // K1
    std::vector<float> V = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,                       // V0
        -1, -2, -3, -4, 2, 3, 4, 5, 6, 7, 8, 9};                     // V1
    std::vector<float> want = {
        5.71f, 6.71f, 7.71f, 8.71f, 5.71f, 6.71f, 7.71f, 8.71f, 5.71f, 6.71f, 7.71f, 8.71f,
        6.07f, 7.07f, 8.07f, 9.07f, 5.00f, 6.00f, 7.00f, 8.00f, 5.71f, 6.71f, 7.71f, 8.71f,
        2.24f, 2.76f, 3.27f, 3.79f, 3.96f, 4.70f, 5.44f, 6.17f, 2.40f, 2.60f, 2.79f, 2.98f,
        0.76f, 0.58f, 0.40f, 0.22f, 1.17f, 1.08f, 1.00f, 0.91f, 2.84f, 3.37f, 3.91f, 4.44f};

    float* dQ = to_dev(Q);
    float* dK = to_dev(K);
    float* dV = to_dev(V);
    float* dO = dev_alloc<float>(48);
    solve(dQ, dK, dV, dO, 4, 2, 3, 4);
    return check_close(from_dev(dO, 48), want, 0.02f, 0.01f);
}
