#include "common.h"

extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP,
                      int S_ROW, int E_ROW, int S_COL, int E_COL);

int main() {
    // N=2, M=2, K=3
    std::vector<int> in = {1, 2, 3, 4, 5, 1, 1, 1, 1, 2, 2, 2};
    int* d_in = to_dev(in);
    int* d_out = dev_alloc<int>(1);
    solve(d_in, d_out, 2, 2, 3, 0, 1, 0, 0, 1, 2);
    return check_exact(from_dev(d_out, 1), std::vector<int>{7});
}
