#include "common.h"

extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations);

int main() {
    // Two obvious clusters; converges in one iteration to centroids (1.5,1.5), (8.5,8.5)
    std::vector<float> x = {1, 2, 8, 9}, y = {1, 2, 8, 9};
    std::vector<float> cx = {1, 8}, cy = {1, 8};
    float* dx = to_dev(x);
    float* dy = to_dev(y);
    float* dcx = to_dev(cx);
    float* dcy = to_dev(cy);
    int* d_labels = dev_alloc<int>(4);
    float* dfx = dev_alloc<float>(2);
    float* dfy = dev_alloc<float>(2);
    solve(dx, dy, d_labels, dcx, dcy, dfx, dfy, 4, 2, 10);

    if (check_exact(from_dev(d_labels, 4), std::vector<int>{0, 0, 1, 1})) return 1;
    if (check_close(from_dev(dfx, 2), {1.5f, 8.5f})) return 1;
    return check_close(from_dev(dfy, 2), {1.5f, 8.5f});
}
