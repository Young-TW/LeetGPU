#include "common.h"

extern "C" void solve(const unsigned int* input, unsigned int* output, int N);

int main() {
    std::vector<unsigned int> in = {170, 45, 75, 90, 2, 802, 24, 66};
    unsigned int* d_in = to_dev(in);
    unsigned int* d_out = dev_alloc<unsigned int>(8);
    solve(d_in, d_out, 8);
    std::vector<unsigned int> want = {2, 24, 45, 66, 75, 90, 170, 802};
    if (check_exact(from_dev(d_out, 8), want)) return 1;

    std::vector<unsigned int> in2 = {1, 4, 1, 3, 555, 1000, 2};
    unsigned int* d_in2 = to_dev(in2);
    unsigned int* d_out2 = dev_alloc<unsigned int>(7);
    solve(d_in2, d_out2, 7);
    std::vector<unsigned int> want2 = {1, 1, 2, 3, 4, 555, 1000};
    return check_exact(from_dev(d_out2, 7), want2);
}
