#include "common.h"

extern "C" void solve(unsigned char* image, int width, int height);

int main() {
    std::vector<unsigned char> img = {255, 0, 128, 255, 0, 255, 0, 255};
    unsigned char* d = to_dev(img);
    solve(d, 1, 2);
    std::vector<unsigned char> want = {0, 255, 127, 255, 255, 0, 255, 255};
    return check_exact(from_dev(d, 8), want);
}
