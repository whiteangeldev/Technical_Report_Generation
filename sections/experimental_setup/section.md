# Experimental Setup

This section specifies the hardware platform, software environment, models, datasets, evaluation metrics, and experimental procedure used to measure the performance of the KV-cache optimizations described in the Implementation section.

## Hardware
All experiments were executed on a single NVIDIA A100 80 GB SXM GPU (Ampere architecture, 2 TB/s memory bandwidth). Host memory was 512 GB DDR4 and the interconnect was PCIe 4.0. Power and thermal limits were left at default BIOS settings; no MIG partitioning or NVLink was used.

## Software Stack
The evaluation used PyTorch 2.1.2 with CUDA 12.1, Hugging Face Transformers 4.38, and a custom KV-cache manager extending the standard `transformers` generation loop. Baselines comprised:
- Full-tensor KV cache (vanilla `transformers` implementation)
- vLLM 0.3.3 with PagedAttention
- H2O and StreamingLLM reference implementations

All kernels were compiled with the same `-O3` flags and used the same cuBLAS/cuDNN back-ends.

## Models and Baselines
Experiments were performed on LLaMA-2-7B and LLaMA-2-13B (both in FP16). Batch size was treated as an independent variable and swept over the discrete set {1, 2, 4, 8, 16, 32}. Context length was fixed at 4 096 tokens for all latency and memory measurements unless otherwise noted.

## Dataset
All reported latency and memory figures were obtained on the LongBench benchmark (Bai et al., 2023) using the “narrativeqa” and “qasper” subsets. Perplexity checks were additionally performed on the WikiText-103 validation set to verify generation quality after eviction and quantization.

## Metrics
- **Peak device memory**: maximum allocated CUDA memory (MiB) measured via `torch.cuda.max_memory_allocated` after the first generated token.
- **Time-to-first-token (TTFT)** and **per-token latency**: wall-clock time (ms) averaged over 128 generated tokens, excluding prompt encoding.
- **Cache size**: number of retained KV entries after eviction, expressed as a percentage of the full-tensor baseline.
- **Quality**: perplexity on WikiText-103 and F1/ROUGE scores on LongBench.

## Procedure
For each model–batch-size pair the following steps were executed:
1. Load model weights and allocate a contiguous KV buffer sized for the maximum context.
2. Warm up the GPU with 10 forward passes.
3. Run 128-token generation on 50 randomly sampled LongBench prompts (seed = 42, temperature = 0.0) while recording memory and latency.
4. Repeat the identical workload for each baseline and for the proposed selective-eviction + layer-wise quantization variant (heavy-hitter ratio = 0.2, sliding-window size = 64; see Principles section for policy details).
5. All timing numbers were averaged over three independent runs; standard deviation is reported in the Results section.
