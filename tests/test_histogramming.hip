#include "common.h"

extern "C" void solve(const int* input, int* histogram, int N, int num_bins);

int main() {
    std::vector<int> in = {0, 1, 2, 1, 0};
    int* d_in = to_dev(in);
    int* d_hist = dev_alloc<int>(3);
    solve(d_in, d_hist, 5, 3);
    if (check_exact(from_dev(d_hist, 3), std::vector<int>{2, 2, 1})) return 1;

    std::vector<int> in2 = {3, 3, 3, 3};
    int* d_in2 = to_dev(in2);
    int* d_hist2 = dev_alloc<int>(5);
    solve(d_in2, d_hist2, 4, 5);
    return check_exact(from_dev(d_hist2, 5), std::vector<int>{0, 0, 0, 4, 0});
}
