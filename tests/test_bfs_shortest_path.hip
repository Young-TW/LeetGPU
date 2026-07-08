#include "common.h"

extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row,
                      int start_col, int end_row, int end_col);

int main() {
    std::vector<int> grid = {0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0};  // 4x4
    int* d_grid = to_dev(grid);
    int* d_res = dev_alloc<int>(1);
    solve(d_grid, d_res, 4, 4, 0, 0, 3, 3);
    if (check_exact(from_dev(d_res, 1), std::vector<int>{6})) return 1;

    std::vector<int> grid2 = {0, 1, 0, 1, 1, 1, 0, 0, 0};  // 3x3, blocked
    int* d_grid2 = to_dev(grid2);
    int* d_res2 = dev_alloc<int>(1);
    solve(d_grid2, d_res2, 3, 3, 0, 0, 0, 2);
    return check_exact(from_dev(d_res2, 1), std::vector<int>{-1});
}
