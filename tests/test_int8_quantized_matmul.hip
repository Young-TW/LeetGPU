#include "common.h"
#include <stdint.h>

extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C);

int main() {
    // Example 2 (example 1 in the spec is inconsistent with the stated formula):
    // A = [[1, 2]], B = [[3], [4]], scales 1, zp (1, 3, 5) -> C = [[6]]
    std::vector<int8_t> A = {1, 2}, B = {3, 4};
    int8_t* dA = to_dev(A);
    int8_t* dB = to_dev(B);
    int8_t* dC = dev_alloc<int8_t>(1);
    solve(dA, dB, dC, 1, 1, 2, 1.0f, 1.0f, 1.0f, 1, 3, 5);
    return check_exact(from_dev(dC, 1), std::vector<int8_t>{6});
}
