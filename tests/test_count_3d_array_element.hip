#include "common.h"

extern "C" void solve(const int* input, int* output, int N, int M, int K, int P);

int main() {
    // N=2, M=2, K=3, p=1 -> 5
    std::vector<int> in = {1, 2, 3, 4, 5, 1, 1, 1, 1, 2, 2, 2};
    int* d_in = to_dev(in);
    int* d_out = dev_alloc<int>(1);
    solve(d_in, d_out, 2, 2, 3, 1);
    return check_exact(from_dev(d_out, 1), std::vector<int>{5});
}
