#include "common.h"

extern "C" void solve(float* data, int N);

int main() {
    std::vector<float> data = {5, 2, 8, 1, 9, 4};
    float* d = to_dev(data);
    solve(d, 6);
    return check_close(from_dev(d, 6), {1, 2, 4, 5, 8, 9});
}
