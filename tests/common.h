#pragma once
// Shared helpers for the local GPU test harnesses. Each test builds the
// example from the problem spec, calls solve(), and checks the output.
//
// Compiles under both runtimes:
//   ROCm: hipcc  (tests/run_tests.sh, links ROCm/*.hip)
//   CUDA: nvcc   (tests/run_tests_cuda.sh, links src/*.cu) — hip* calls are
//                mapped to the cuda* equivalents below.
#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipMemset cudaMemset
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                              \
    do {                                                                          \
        hipError_t err = (x);                                                     \
        if (err != hipSuccess) {                                                  \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err),    \
                    __FILE__, __LINE__);                                          \
            exit(2);                                                              \
        }                                                                         \
    } while (0)

template <typename T>
T* to_dev(const std::vector<T>& v) {
    T* d = nullptr;
    HIP_CHECK(hipMalloc(&d, v.size() * sizeof(T)));
    HIP_CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

template <typename T>
T* dev_alloc(size_t n) {
    T* d = nullptr;
    HIP_CHECK(hipMalloc(&d, n * sizeof(T)));
    HIP_CHECK(hipMemset(d, 0, n * sizeof(T)));
    return d;
}

template <typename T>
std::vector<T> from_dev(const T* d, size_t n) {
    std::vector<T> v(n);
    HIP_CHECK(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

inline bool nearly_equal(float got, float want, float atol, float rtol) {
    if (std::isnan(got) != std::isnan(want)) return false;
    if (std::isnan(got)) return true;
    if (std::isinf(want) || std::isinf(got)) return got == want;
    return std::fabs(got - want) <= atol + rtol * std::fabs(want);
}

inline int check_close(const std::vector<float>& got, const std::vector<float>& want,
                       float atol = 1e-3f, float rtol = 1e-3f) {
    if (got.size() != want.size()) {
        printf("FAIL size mismatch: got %zu want %zu\n", got.size(), want.size());
        return 1;
    }
    for (size_t i = 0; i < got.size(); i++) {
        if (!nearly_equal(got[i], want[i], atol, rtol)) {
            printf("FAIL at [%zu]: got %g want %g\n", i, got[i], want[i]);
            return 1;
        }
    }
    printf("PASS\n");
    return 0;
}

template <typename T>
inline int check_exact(const std::vector<T>& got, const std::vector<T>& want) {
    if (got.size() != want.size()) {
        printf("FAIL size mismatch: got %zu want %zu\n", got.size(), want.size());
        return 1;
    }
    for (size_t i = 0; i < got.size(); i++) {
        if (got[i] != want[i]) {
            printf("FAIL at [%zu]: got %lld want %lld\n", i, (long long)got[i],
                   (long long)want[i]);
            return 1;
        }
    }
    printf("PASS\n");
    return 0;
}
