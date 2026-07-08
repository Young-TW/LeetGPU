# LeetGPU

Solutions to [LeetGPU](https://leetgpu.com) challenges, written in CUDA.

- `problems/` — challenge statements fetched from leetgpu.com
- `src/` — solutions (CUDA `.cu`; #41 Simple Inference is PyTorch-only)
- `ROCm/` — HIP ports of the solutions (generated with `hipify-perl`)
- `tests/` — local test harnesses built from each spec's examples
- `tools/` — fetch / compile-check / remote-run scripts

## Local verification (CUDA or ROCm)

Every problem has a harness in `tests/` that feeds the spec's example inputs to
`solve()` on your own GPU and checks the output — see
[tests/README.md](tests/README.md) for details.

```sh
# AMD GPU (ROCm / hipcc), tests link ROCm/*.hip
tests/run_tests.sh                    # all problems
GPU_ARCH=gfx1100 tests/run_tests.sh   # non-default arch (default gfx1201)

# NVIDIA GPU (CUDA / nvcc), tests link src/*.cu
tests/run_tests_cuda.sh
NVCC_ARCH="-arch=sm_86" tests/run_tests_cuda.sh

# single problem, either runtime
tests/run_tests.sh softmax
tests/run_tests_cuda.sh softmax

# 41 Simple Inference (PyTorch)
python3 tests/test_simple_inference.py
```

Status: all 88 GPU tests + the PyTorch test pass on ROCm (RX 9070 XT, gfx1201).
These are the small spec examples only — LeetGPU's official judge additionally
runs large hidden tests and performance measurement.

## Problems

| # | Title | Difficulty | Problem | Solution |
|---|-------|------------|---------|----------|
| 1 | Vector Addition | easy | [spec](problems/001_vector_addition.md) | [src](src/vector_addition.cu) |
| 2 | Matrix Multiplication | easy | [spec](problems/002_matrix_multiplication.md) | [src](src/matrix_multiplication.cu) |
| 3 | Matrix Transpose | easy | [spec](problems/003_matrix_transpose.md) | [src](src/matrix_transpose.cu) |
| 4 | Reduction | medium | [spec](problems/004_reduction.md) | [src](src/reduction.cu) |
| 5 | Softmax | medium | [spec](problems/005_softmax.md) | [src](src/softmax.cu) |
| 6 | Softmax Attention | medium | [spec](problems/006_softmax_attention.md) | [src](src/softmax_attention.cu) |
| 7 | Color Inversion | easy | [spec](problems/007_color_inversion.md) | [src](src/color_inversion.cu) |
| 8 | Matrix Addition | easy | [spec](problems/008_matrix_addition.md) | [src](src/matrix_addition.cu) |
| 9 | 1D Convolution | easy | [spec](problems/009_1d_convolution.md) | [src](src/1d_convolution.cu) |
| 10 | 2D Convolution | medium | [spec](problems/010_2d_convolution.md) | [src](src/2d_convolution.cu) |
| 11 | 3D Convolution | medium | [spec](problems/011_3d_convolution.md) | [src](src/3d_convolution.cu) |
| 12 | Multi-Head Attention | hard | [spec](problems/012_multi_head_attention.md) | [src](src/multi_head_attention.cu) |
| 13 | Histogramming | medium | [spec](problems/013_histogramming.md) | [src](src/histogramming.cu) |
| 14 | Multi-Agent Simulation | hard | [spec](problems/014_multi_agent_simulation.md) | [src](src/multi_agent_simulation.cu) |
| 15 | Sorting | hard | [spec](problems/015_sorting.md) | [src](src/sorting.cu) |
| 16 | Prefix Sum | medium | [spec](problems/016_prefix_sum.md) | [src](src/prefix_sum.cu) |
| 17 | Dot Product | medium | [spec](problems/017_dot_product.md) | [src](src/dot_product.cu) |
| 18 | Sparse Matrix-Vector Multiplication | medium | [spec](problems/018_sparse_matrix_vector_multiplication.md) | [src](src/sparse_matrix_vector_multiplication.cu) |
| 19 | Reverse Array | easy | [spec](problems/019_reverse_array.md) | [src](src/reverse_array.cu) |
| 20 | K-Means Clustering | hard | [spec](problems/020_k_means_clustering.md) | [src](src/k_means_clustering.cu) |
| 21 | ReLU | easy | [spec](problems/021_relu.md) | [src](src/relu.cu) |
| 22 | General Matrix Multiplication (GEMM) | medium | [spec](problems/022_general_matrix_multiplication_gemm.md) | [src](src/general_matrix_multiplication_gemm.cu) |
| 23 | Leaky ReLU | easy | [spec](problems/023_leaky_relu.md) | [src](src/leaky_relu.cu) |
| 24 | Rainbow Table | easy | [spec](problems/024_rainbow_table.md) | [src](src/rainbow_table.cu) |
| 25 | Categorical Cross Entropy Loss | medium | [spec](problems/025_categorical_cross_entropy_loss.md) | [src](src/categorical_cross_entropy_loss.cu) |
| 27 | Mean Squared Error | medium | [spec](problems/027_mean_squared_error.md) | [src](src/mean_squared_error.cu) |
| 28 | Gaussian Blur | medium | [spec](problems/028_gaussian_blur.md) | [src](src/gaussian_blur.cu) |
| 29 | Top K Selection | medium | [spec](problems/029_top_k_selection.md) | [src](src/top_k_selection.cu) |
| 30 | Batched Matrix Multiplication | medium | [spec](problems/030_batched_matrix_multiplication.md) | [src](src/batched_matrix_multiplication.cu) |
| 31 | Matrix Copy | easy | [spec](problems/031_matrix_copy.md) | [src](src/matrix_copy.cu) |
| 32 | INT8 Quantized MatMul | medium | [spec](problems/032_int8_quantized_matmul.md) | [src](src/int8_quantized_matmul.cu) |
| 33 | Ordinary Least Squares | medium | [spec](problems/033_ordinary_least_squares.md) | [src](src/ordinary_least_squares.cu) |
| 34 | Logistic Regression | medium | [spec](problems/034_logistic_regression.md) | [src](src/logistic_regression.cu) |
| 35 | Monte Carlo Integration | medium | [spec](problems/035_monte_carlo_integration.md) | [src](src/monte_carlo_integration.cu) |
| 36 | Radix Sort | hard | [spec](problems/036_radix_sort.md) | [src](src/radix_sort.cu) |
| 37 | Matrix Power | medium | [spec](problems/037_matrix_power.md) | [src](src/matrix_power.cu) |
| 38 | Nearest Neighbor | medium | [spec](problems/038_nearest_neighbor.md) | [src](src/nearest_neighbor.cu) |
| 39 | Fast Fourier Transform | hard | [spec](problems/039_fast_fourier_transform.md) | [src](src/fast_fourier_transform.cu) |
| 40 | Batch Normalization | medium | [spec](problems/040_batch_normalization.md) | [src](src/batch_normalization.cu) |
| 41 | Simple Inference | easy | [spec](problems/041_simple_inference.md) | [src](src/simple_inference.py) |
| 42 | 2D Max Pooling | medium | [spec](problems/042_2d_max_pooling.md) | [src](src/2d_max_pooling.cu) |
| 43 | Count Array Element | medium | [spec](problems/043_count_array_element.md) | [src](src/count_array_element.cu) |
| 44 | Count 2D Array Element | medium | [spec](problems/044_count_2d_array_element.md) | [src](src/count_2d_array_element.cu) |
| 45 | Count 3D Array Element | medium | [spec](problems/045_count_3d_array_element.md) | [src](src/count_3d_array_element.cu) |
| 46 | BFS Shortest Path | hard | [spec](problems/046_bfs_shortest_path.md) | [src](src/bfs_shortest_path.cu) |
| 47 | Subarray Sum | medium | [spec](problems/047_subarray_sum.md) | [src](src/subarray_sum.cu) |
| 48 | 2D Subarray Sum | medium | [spec](problems/048_2d_subarray_sum.md) | [src](src/2d_subarray_sum.cu) |
| 49 | 3D Subarray Sum | medium | [spec](problems/049_3d_subarray_sum.md) | [src](src/3d_subarray_sum.cu) |
| 50 | RMS Normalization | medium | [spec](problems/050_rms_normalization.md) | [src](src/rms_normalization.cu) |
| 51 | Max Subarray Sum | medium | [spec](problems/051_max_subarray_sum.md) | [src](src/max_subarray_sum.cu) |
| 52 | Sigmoid Linear Unit | easy | [spec](problems/052_sigmoid_linear_unit.md) | [src](src/sigmoid_linear_unit.cu) |
| 53 | Causal Self-Attention | hard | [spec](problems/053_causal_self_attention.md) | [src](src/causal_self_attention.cu) |
| 54 | Swish-Gated Linear Unit | easy | [spec](problems/054_swish_gated_linear_unit.md) | [src](src/swish_gated_linear_unit.cu) |
| 55 | Attention with Linear Biases | medium | [spec](problems/055_attention_with_linear_biases.md) | [src](src/attention_with_linear_biases.cu) |
| 56 | Linear Self-Attention | hard | [spec](problems/056_linear_self_attention.md) | [src](src/linear_self_attention.cu) |
| 57 | FP16 Batched Matrix Multiplication | medium | [spec](problems/057_fp16_batched_matrix_multiplication.md) | [src](src/fp16_batched_matrix_multiplication.cu) |
| 58 | FP16 Dot Product | medium | [spec](problems/058_fp16_dot_product.md) | [src](src/fp16_dot_product.cu) |
| 59 | Sliding Window Self-Attention | hard | [spec](problems/059_sliding_window_self_attention.md) | [src](src/sliding_window_self_attention.cu) |
| 60 | Top-p Sampling | medium | [spec](problems/060_top_p_sampling.md) | [src](src/top_p_sampling.cu) |
| 61 | Rotary Positional Embedding | medium | [spec](problems/061_rotary_positional_embedding.md) | [src](src/rotary_positional_embedding.cu) |
| 62 | Value Clipping | easy | [spec](problems/062_value_clipping.md) | [src](src/value_clipping.cu) |
| 63 | Interleave Arrays | easy | [spec](problems/063_interleave_arrays.md) | [src](src/interleave_arrays.cu) |
| 64 | Weight Dequantization | medium | [spec](problems/064_weight_dequantization.md) | [src](src/weight_dequantization.cu) |
| 65 | Gaussian Error Gated Linear Unit | easy | [spec](problems/065_gaussian_error_gated_linear_unit.md) | [src](src/gaussian_error_gated_linear_unit.cu) |
| 66 | RGB to Grayscale | easy | [spec](problems/066_rgb_to_grayscale.md) | [src](src/rgb_to_grayscale.cu) |
| 67 | MoE Top-K Gating | medium | [spec](problems/067_moe_top_k_gating.md) | [src](src/moe_top_k_gating.cu) |
| 68 | Sigmoid Activation | easy | [spec](problems/068_sigmoid_activation.md) | [src](src/sigmoid_activation.cu) |
| 69 | 2D Jacobi Stencil | medium | [spec](problems/069_2d_jacobi_stencil.md) | [src](src/2d_jacobi_stencil.cu) |
| 70 | Segmented Exclusive Prefix Sum | medium | [spec](problems/070_segmented_exclusive_prefix_sum.md) | [src](src/segmented_exclusive_prefix_sum.cu) |
| 71 | Parallel Merge | medium | [spec](problems/071_parallel_merge.md) | [src](src/parallel_merge.cu) |
| 72 | Stream Compaction | medium | [spec](problems/072_stream_compaction.md) | [src](src/stream_compaction.cu) |
| 73 | All-Pairs Shortest Paths | hard | [spec](problems/073_all_pairs_shortest_paths.md) | [src](src/all_pairs_shortest_paths.cu) |
| 74 | GPT-2 Transformer Block | hard | [spec](problems/074_gpt_2_transformer_block.md) | [src](src/gpt_2_transformer_block.cu) |
| 75 | Sparse Matrix-Dense Matrix Multiplication | medium | [spec](problems/075_sparse_matrix_dense_matrix_multiplication.md) | [src](src/sparse_matrix_dense_matrix_multiplication.cu) |
| 76 | Adder Transformer Inference | medium | [spec](problems/076_adder_transformer_inference.md) | [src](src/adder_transformer_inference.cu) |
| 78 | 2D FFT | medium | [spec](problems/078_2d_fft.md) | [src](src/2d_fft.cu) |
| 80 | Grouped Query Attention | medium | [spec](problems/080_grouped_query_attention.md) | [src](src/grouped_query_attention.cu) |
| 81 | INT4 Weight-Only Quantized MatMul | medium | [spec](problems/081_int4_weight_only_quantized_matmul.md) | [src](src/int4_weight_only_quantized_matmul.cu) |
| 82 | Linear Recurrence | medium | [spec](problems/082_linear_recurrence.md) | [src](src/linear_recurrence.cu) |
| 84 | SwiGLU MLP Block | medium | [spec](problems/084_swiglu_mlp_block.md) | [src](src/swiglu_mlp_block.cu) |
| 85 | LoRA Linear | medium | [spec](problems/085_lora_linear.md) | [src](src/lora_linear.cu) |
| 87 | Speculative Decoding Verification | medium | [spec](problems/087_speculative_decoding_verification.md) | [src](src/speculative_decoding_verification.cu) |
| 90 | Causal Depthwise Conv1d | medium | [spec](problems/090_causal_depthwise_conv1d.md) | [src](src/causal_depthwise_conv1d.cu) |
| 92 | Decaying Causal Attention | medium | [spec](problems/092_decaying_causal_attention.md) | [src](src/decaying_causal_attention.cu) |
| 93 | Llama Transformer Block | hard | [spec](problems/093_llama_transformer_block.md) | [src](src/llama_transformer_block.cu) |
| 94 | SSM Selective Scan | medium | [spec](problems/094_ssm_selective_scan.md) | [src](src/ssm_selective_scan.cu) |
| 96 | INT8 KV-Cache Attention | medium | [spec](problems/096_int8_kv_cache_attention.md) | [src](src/int8_kv_cache_attention.cu) |
| 105 | Group Normalization | medium | [spec](problems/105_group_normalization.md) | [src](src/group_normalization.cu) |
| 106 | Token Embedding Layer | medium | [spec](problems/106_token_embedding_layer.md) | [src](src/token_embedding_layer.cu) |
