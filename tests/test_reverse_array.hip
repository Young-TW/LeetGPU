#include "common.h"

extern "C" void solve(float* input, int N);

int main() {
    std::vector<float> a = {1, 2, 3, 4};
    float* d = to_dev(a);
    solve(d, 4);
    if (check_close(from_dev(d, 4), {4, 3, 2, 1})) return 1;

    std::vector<float> b = {1.5f, 2.5f, 3.5f};
    float* d2 = to_dev(b);
    solve(d2, 3);
    return check_close(from_dev(d2, 3), {3.5f, 2.5f, 1.5f});
}
