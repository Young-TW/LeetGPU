#include <hip/hip_runtime.h>
#include <cstdio>

extern "C" void solve(const float* A, const float* B, float* C, int N);

int main() {
    // Example 1 from the Vector Addition spec
    float hA[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hB[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float expected[] = {6.0f, 8.0f, 10.0f, 12.0f};
    int N = 4;

    float *dA, *dB, *dC;
    hipMalloc(&dA, N * sizeof(float));
    hipMalloc(&dB, N * sizeof(float));
    hipMalloc(&dC, N * sizeof(float));
    hipMemcpy(dA, hA, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, N * sizeof(float), hipMemcpyHostToDevice);

    solve(dA, dB, dC, N);

    float hC[4];
    hipMemcpy(hC, dC, N * sizeof(float), hipMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        printf("C[%d] = %g (expected %g)\n", i, hC[i], expected[i]);
        if (hC[i] != expected[i]) ok = false;
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
