#include "common.h"
#include <stdint.h>

extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim);

int main() {
    // num_heads=1, seq_len=3, head_dim=4
    std::vector<float> Q = {1, 1, 1, 1};
    std::vector<int8_t> K = {10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0};
    std::vector<int8_t> V = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    std::vector<float> ks = {0.1f, 0.1f, 0.1f}, vs = {0.1f, 0.1f, 0.1f};
    float* dQ = to_dev(Q);
    int8_t* dK = to_dev(K);
    int8_t* dV = to_dev(V);
    float* dks = to_dev(ks);
    float* dvs = to_dev(vs);
    float* dO = dev_alloc<float>(4);
    solve(dQ, dK, dV, dks, dvs, dO, 1, 3, 4);
    return check_close(from_dev(dO, 4), {5, 6, 7, 8}, 0.01f, 0.01f);
}
