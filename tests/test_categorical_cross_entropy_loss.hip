#include "common.h"

extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C);

int main() {
    std::vector<float> logits = {1.0f, 2.0f, 0.5f, 0.1f, 3.0f, 1.5f};  // 2x3
    std::vector<int> labels = {1, 1};
    float* d_logits = to_dev(logits);
    int* d_labels = to_dev(labels);
    float* d_loss = dev_alloc<float>(1);
    solve(d_logits, d_labels, d_loss, 2, 3);
    if (check_close(from_dev(d_loss, 1), {0.3548926f})) return 1;

    std::vector<float> l2 = {-0.5f, 1.5f, 0.0f, 1.0f, 2.0f, -1.0f,
                             0.5f,  0.5f, 0.0f, 0.0f, 0.0f, 0.0f};  // 3x4
    std::vector<int> lab2 = {3, 0, 1};
    float* d_l2 = to_dev(l2);
    int* d_lab2 = to_dev(lab2);
    float* d_loss2 = dev_alloc<float>(1);
    solve(d_l2, d_lab2, d_loss2, 3, 4);
    return check_close(from_dev(d_loss2, 1), {0.98820376f});
}
