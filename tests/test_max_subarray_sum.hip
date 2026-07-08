#include "common.h"

extern "C" void solve(const int* input, int* output, int N, int window_size);

int main() {
    std::vector<int> in = {1, 2, 4, 2, 3};
    int* d_in = to_dev(in);
    int* d_out = dev_alloc<int>(1);
    solve(d_in, d_out, 5, 2);
    if (check_exact(from_dev(d_out, 1), std::vector<int>{6})) return 1;

    std::vector<int> in2 = {-1, -4, -2, 1};
    int* d_in2 = to_dev(in2);
    int* d_out2 = dev_alloc<int>(1);
    solve(d_in2, d_out2, 4, 3);
    return check_exact(from_dev(d_out2, 1), std::vector<int>{-5});
}
