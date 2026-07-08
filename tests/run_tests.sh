#!/usr/bin/env bash
# Build every tests/test_<slug>.hip against ROCm/<slug>.hip and run on the GPU.
# Usage: tests/run_tests.sh [slug ...]
set -u
cd "$(dirname "$0")/.."

ARCH="${GPU_ARCH:-gfx1201}"
BUILD=tests/build
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

# compile in parallel
printf '%s\n' "${slugs[@]}" | xargs -P 8 -I{} sh -c '
    if ! /opt/rocm/bin/hipcc tests/test_{}.hip ROCm/{}.hip -o tests/build/{} \
         --offload-arch='"$ARCH"' -w -Itests 2> tests/build/{}.cerr; then
        echo COMPILE_FAIL > tests/build/{}.result
    fi'

pass=0; fail=0
for s in "${slugs[@]}"; do
    if [ -f "$BUILD/$s.result" ]; then
        echo "[$s] COMPILE FAIL"
        sed -n '1,5p' "$BUILD/$s.cerr"
        fail=$((fail+1)); rm -f "$BUILD/$s.result"
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
