#include "common.h"

extern "C" void solve(const int* input, unsigned int* output, int N, int R);

int main() {
    std::vector<int> in = {123, 456, 789};
    int* d_in = to_dev(in);
    unsigned int* d_out = dev_alloc<unsigned int>(3);
    solve(d_in, d_out, 3, 2);
    std::vector<unsigned int> want = {1636807824u, 1273011621u, 2193987222u};
    if (check_exact(from_dev(d_out, 3), want)) return 1;

    std::vector<int> in2 = {0, 1, 2147483647};
    int* d_in2 = to_dev(in2);
    unsigned int* d_out2 = dev_alloc<unsigned int>(3);
    solve(d_in2, d_out2, 3, 3);
    std::vector<unsigned int> want2 = {96754810u, 3571711400u, 2006156166u};
    return check_exact(from_dev(d_out2, 3), want2);
}
