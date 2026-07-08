#include "common.h"

extern "C" void solve(const float* values, const int* flags, float* output, int N);

int main() {
    std::vector<float> vals = {1, 2, 3, 4, 5, 6};
    std::vector<int> flags = {1, 0, 0, 1, 0, 1};
    float* d_v = to_dev(vals);
    int* d_f = to_dev(flags);
    float* d_out = dev_alloc<float>(6);
    solve(d_v, d_f, d_out, 6);
    return check_close(from_dev(d_out, 6), {0, 1, 3, 0, 4, 0});
}
