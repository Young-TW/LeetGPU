# tests/ — 本地驗證範例測資

每題一個 harness(`test_<slug>.hip`),用題目規格(`problems/*.md`)裡的 Example
當測資:在 host 端準備輸入 → 搬上 GPU → 呼叫該題的 `solve()` → 搬回來與期望輸出比對,
印出 `PASS` 或第一個不符的位置。

harness 是 runtime 中立的:`common.h` 在 hipcc 下用 HIP API,在 nvcc 下自動把
`hip*` 對應到 `cuda*`,所以同一份測試可以驗 `ROCm/*.hip` 也可以驗 `src/*.cu`。

## ROCm(AMD GPU)

需求:ROCm(`/opt/rocm/bin/hipcc`)。

```sh
# 全部跑一遍(先平行編譯,再逐一在 GPU 上執行)
tests/run_tests.sh

# 只跑指定題目(用 slug,即 src/ 檔名去掉副檔名)
tests/run_tests.sh softmax reduction top_k_selection

# 非預設 GPU 架構(預設 gfx1201;用 `rocminfo | grep gfx` 查你的)
GPU_ARCH=gfx1100 tests/run_tests.sh
```

編譯產物在 `tests/build/`(已 gitignore)。測試連結的是 `ROCm/<slug>.hip`。

## CUDA(NVIDIA GPU)

需求:CUDA Toolkit(`nvcc`)。

```sh
# 全部跑一遍,直接連結原始 CUDA 解答 src/<slug>.cu
tests/run_tests_cuda.sh

# 指定架構或 nvcc 路徑
NVCC_ARCH="-arch=sm_86" tests/run_tests_cuda.sh
NVCC=/usr/local/cuda/bin/nvcc tests/run_tests_cuda.sh softmax
```

編譯產物在 `tests/build-cuda/`。

## \#41 Simple Inference(PyTorch)

這題只有 PyTorch 版本,用對應平台的 PyTorch(CUDA 或 ROCm 版)執行:

```sh
python3 tests/test_simple_inference.py
```

## 手動驗證單一題目

不想用 runner 的話,pattern 很簡單(見任何一個 `test_*.hip`):

```sh
# ROCm
hipcc tests/test_softmax.hip ROCm/softmax.hip -o /tmp/t --offload-arch=gfx1201 -w -Itests && /tmp/t

# CUDA
nvcc -x cu tests/test_softmax.hip src/softmax.cu -o /tmp/t -Itests -w && /tmp/t
```

## 測資範圍與已知規格問題

- 86/89 題用規格 Example 的具體數值驗證。
- 3 題規格沒有具體測資,改用合成測資:
  - **#20 K-Means**:雙聚類資料,收斂結果可手算
  - **#74 GPT-2 Block、#93 Llama Block**:全零權重(兩個子層輸出為 0,殘差使 output == x),
    驗證整條管線與權重 offset
- 規格本身有三處 typo(測試裡以修正值驗證並附註):
  - **#53** Example 2 第二列應為 `[4, 5]`(等分數 softmax 必為均分)
  - **#55** Example 1 第一列應為 `[3.05, 4.05, 5.05, 6.05]`(凸組合必為等差)
  - **#32** Example 1 與題目公式矛盾(改用 Example 2)
- 本地驗的是小規模範例;LeetGPU 官方 judge 另有大規模測資與效能門檻,
  線上驗證見 `tools/README.md`。

## ROCm/ 是怎麼來的

`ROCm/*.hip` 由 `hipify-perl` 從 `src/*.cu` 轉換。改了 `src/` 之後重新產生:

```sh
for f in src/*.cu; do
    /opt/rocm/bin/hipify-perl "$f" > "ROCm/$(basename "$f" .cu).hip"
done
```
