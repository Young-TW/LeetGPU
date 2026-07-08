#!/usr/bin/env bash
# CUDA variant of run_tests.sh: build every tests/test_<slug>.hip against the
# original CUDA solution src/<slug>.cu with nvcc, then run on the GPU.
# Usage: tests/run_tests_cuda.sh [slug ...]
set -u
cd "$(dirname "$0")/.."

NVCC="${NVCC:-nvcc}"
NVCC_ARCH="${NVCC_ARCH:-}"   # e.g. NVCC_ARCH="-arch=sm_86"; empty lets nvcc pick
BUILD=tests/build-cuda
mkdir -p "$BUILD"

slugs=()
if [ $# -gt 0 ]; then
    slugs=("$@")
else
    for t in tests/test_*.hip; do
        s=$(basename "$t" .hip)
        slugs+=("${s#test_}")
    done
fi

pass=0; fail=0
for s in "${slugs[@]}"; do
    # -x cu compiles the .hip harness as CUDA source; common.h maps hip* -> cuda*
    if ! "$NVCC" $NVCC_ARCH -x cu "tests/test_$s.hip" "src/$s.cu" -o "$BUILD/$s" \
         -Itests -w 2> "$BUILD/$s.cerr"; then
        echo "[$s] COMPILE FAIL"
        sed -n '1,5p' "$BUILD/$s.cerr"
        fail=$((fail+1))
        continue
    fi
    out=$(timeout 60 "$BUILD/$s" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then
        pass=$((pass+1))
    else
        echo "[$s] FAIL (rc=$rc)"
        echo "$out" | head -5
        fail=$((fail+1))
    fi
done
echo "----"
echo "pass: $pass  fail: $fail  total: ${#slugs[@]}"
