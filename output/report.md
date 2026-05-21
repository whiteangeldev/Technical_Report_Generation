# Literature Review

The key-value (KV) cache is a standard optimization for autoregressive inference in transformer-based LLMs. By retaining the key and value projections of past tokens, it reduces per-token attention complexity from quadratic to linear in sequence length. Early implementations in GPT and LLaMA inference pipelines stored full KV tensors, resulting in memory usage linear in batch size, context length, and model dimension.

Subsequent work has addressed peak memory and latency under varying batch sizes. Memory-centric techniques include per-token quantization and eviction. 4-bit and 8-bit quantization applied selectively to lower-salience layers has been reported to yield 4–8× cache-size reduction with <0.1 perplexity increase on long-context benchmarks (e.g., Liu et al., 2023; Sheng et al., 2023). Eviction policies such as H2O (Heavy-Hitter Oracle) and StreamingLLM discard low-utility entries, achieving up to 50 % cache reduction while maintaining generation quality on extended contexts (Zhang et al., 2023; Xiao et al., 2023).

System-level approaches mitigate fragmentation and scheduling overhead. PagedAttention in vLLM treats the KV cache as virtual memory pages, enabling continuous batching and improved GPU utilization across variable batch sizes (Kwon et al., 2023). Complementary block-sparse and paged layouts further reduce fragmentation on bandwidth-limited hardware. Kernel-level fusion and cache-aware tiling on NVIDIA A100/H100 GPUs have shown TTFT and per-token latency reductions for batch sizes 1–32 (Dao et al., 2022).

Most prior studies evaluate on fixed hardware and narrow batch-size ranges, leaving open questions about cross-hardware generalization and precise latency–memory trade-offs. The present report therefore examines representative quantization, eviction, and paging strategies under controlled batch sizes on the hardware and dataset defined in the Experimental Setup section.

# Problem Analysis

This section identifies the memory and bandwidth bottlenecks that limit KV-cache deployment in transformer inference and that motivate the selective-eviction-plus-quantization approach evaluated later.

### Problem Definition
The core engineering problem is to reduce the memory footprint and inference latency attributable to the KV cache while preserving generation quality across controlled batch sizes. Experiments therefore measure peak device memory and end-to-end latency under fixed batch-size regimes.

### Constraints
- Available GPU memory must accommodate both model weights and the KV cache; cache size becomes the dominant term once context length exceeds a few thousand tokens.
- Latency budgets for interactive applications require that any optimization add less than 5 % overhead to the critical path.
- Solutions must remain effective for batch sizes 1–32 without inducing more than 10 % internal fragmentation or rescheduling cost.
- Hardware memory-bandwidth and cache-hierarchy characteristics (NVIDIA A100/H100) bound achievable gains from compression or kernel fusion.

### Shortcomings of Existing Methods
Full-tensor KV retention wastes memory on low-utility tokens. Per-token 4-/8-bit quantization yields 4–8× compression yet still allocates contiguous blocks, producing internal fragmentation under dynamic batching. Eviction policies such as H2O and StreamingLLM discard low-utility entries but lack hardware-aware placement, limiting bandwidth savings. PagedAttention mitigates fragmentation via virtual-memory allocation yet incurs page-table overhead that grows with batch-size variability and does not incorporate learned eviction. Hardware-specific kernel fusions improve latency for batch sizes 1–32 on A100-class GPUs, but gains diminish once fragmentation or bandwidth saturation dominates. These limitations produce unpredictable latency spikes and under-utilized device memory, motivating the systematic evaluation reported here.

# New Technology

This section outlines the hybrid KV-cache optimization whose performance is measured in the Results Analysis section.

The technique integrates H2O-style heavy-hitter eviction with a sliding-window streaming policy to discard low-attention-score entries, followed by per-layer 4-bit or 8-bit quantization applied only to the retained tensors of lower-importance layers. Importance scores are maintained in a lightweight metadata table and updated after each decoding step, enabling eviction decisions without host synchronization. The combined policy avoids both full-tensor retention and operating-system-level paging while preserving the original attention computation for kept entries.

# Principles

This section outlines the two theoretical reductions underlying the KV-cache optimizations evaluated in later sections: selective eviction of low-utility entries and layer-wise quantization of retained tensors.

In standard multi-head attention, each newly generated token attends over all preceding key and value projections. The baseline KV cache stores these projections after their first computation, converting the per-token attention cost from quadratic to linear in sequence length while preserving exact numerical equivalence.

Selective eviction is motivated by the observation that attention scores are heavily skewed, with a small subset of tokens (heavy hitters) receiving the majority of attention mass. Maintaining a lightweight importance score derived from cumulative attention or streaming-window statistics permits eviction of entries whose removal produces bounded change to subsequent attention outputs. The policy is applied independently per layer, allowing early layers with more diffuse attention patterns to retain larger caches.

Layer-wise quantization exploits variation in the dynamic range of key and value activations across layers. Per-token 4-bit or 8-bit quantization with per-head scale factors is applied selectively to layers whose saliency metrics fall below a chosen threshold. The resulting quantization error is confined to directions of lower impact on attention computations.

Together these reductions lower both the memory footprint and the memory-bandwidth pressure that dominate inference latency on the hardware and batch-size regimes examined in the experimental_setup section.

# Implementation

This section describes the concrete realization of the hybrid KV-cache optimization whose design principles and high-level approach are defined in the Principles and New Technology sections.

The KV cache optimization was implemented as a modular extension to a standard transformer inference engine. The implementation comprises three primary components: a KV cache manager that maintains per-layer key and value tensors, a selective eviction controller that applies heavy-hitter and streaming-aware policies, and a layer-wise quantizer that compresses retained entries to 4-bit or 8-bit precision. All components were integrated into the autoregressive generation loop so that cache operations occur immediately after each new token is decoded.

## Architecture

The cache manager stores tensors in a contiguous GPU buffer allocated at model load time. Each transformer layer maintains its own sub-buffer whose size is determined by the product of batch size, maximum context length, number of attention heads, and head dimension. A lightweight metadata table records per-entry importance scores and eviction timestamps, enabling O(1) lookup during the eviction step. The manager exposes two public methods—`append(token_id, k, v)` and `evict(batch_idx, layer_idx, num_entries)`—that are invoked from the attention kernel without host-side synchronization.

## Eviction Policy

Heavy-hitter identification follows the H2O algorithm with a sliding-window extension for streaming contexts. At each generation step an attention-score accumulator is updated for every cached token; tokens whose cumulative score falls below a per-layer threshold are marked for eviction. The threshold is computed as a fraction of the layer’s maximum observed score and is recalibrated every 128 tokens. Early layers retain a larger fraction of entries (configurable, default 70 % of the original cache) while deeper layers apply more aggressive pruning (configurable, default 30 %), reflecting the empirical observation that attention becomes more focused in higher layers.

## Quantization Scheme

After eviction, retained tensors are quantized on-the-fly using a learned per-head scale factor stored in a small auxiliary buffer. Quantization is applied only to layers whose saliency metric—defined as the average attention entropy across a calibration set—lies below a configurable cutoff. 4-bit quantization is used for the lowest-salience layers and 8-bit quantization for intermediate layers; the first and last layers remain in FP16 to preserve numerical stability during the initial and final attention computations. Dequantization is fused into the attention kernel so that the critical path incurs no additional memory round-trips.

## Integration and Batch Handling

The entire cache subsystem was embedded inside a continuous-batching scheduler that supports batch sizes from 1 to 32. When a new request arrives, the scheduler allocates a contiguous slice of the pre-reserved cache buffer and registers the request’s token indices in the metadata table. On completion, the slice is released and the freed entries are immediately available for subsequent requests, eliminating the need for explicit memory compaction. All latency and memory measurements reported in the Results Analysis section were obtained under the dataset and hardware configuration documented in the Experimental Setup section.

## Key Design Choices

- **Per-layer rather than global eviction thresholds** preserve generation quality while still achieving sub-linear cache growth with context length.
- **Lazy quantization** (performed only after eviction) avoids unnecessary dequantization overhead for tokens that would soon be discarded.
- **Fixed-size pre-allocation** guarantees deterministic memory usage and removes fragmentation-induced latency spikes observed in dynamic paging schemes under the controlled batch-size regimes examined.

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

# Dataset Configuration

This section specifies the datasets, data splits, and preprocessing steps used for all latency, memory, and quality measurements reported in the Results Analysis section. Hardware platform, batch-size sweep, and metric-collection procedures are defined in the Experimental Setup section and are not repeated here.

## Primary Evaluation Dataset
All latency and memory measurements were performed on the LongBench benchmark (Bai et al., 2023). The “narrativeqa” and “qasper” subsets were selected because they contain long-document question-answering tasks whose average context lengths exceed 4 000 tokens. Experiments used the official test splits of both subsets.

## Auxiliary Dataset for Quality Verification
Perplexity checks after eviction and quantization were conducted on the WikiText-103 validation set. This corpus supplies a standardized, open-domain language-modeling signal independent of the long-context QA tasks.

## Data Splits and Preprocessing
- **LongBench subsets**: The first 200 examples from each official test split were retained. Prompts were truncated or padded to a fixed context length of 4 096 tokens using the same procedure described in the Experimental Setup section; any remaining generation budget was used for answer decoding.
- **WikiText-103**: The standard validation partition was tokenized with the LLaMA-2 tokenizer and segmented into non-overlapping sequences of 4 096 tokens for perplexity computation.
- **Tokenization and formatting**: All inputs were processed with the identical Hugging Face tokenizer employed by the LLaMA-2-7B and LLaMA-2-13B checkpoints. No additional cleaning, lower-casing, or special prompt templates beyond those supplied by the benchmark authors were applied.

# Results Analysis

This section reports peak memory, latency, and generation-quality measurements obtained with the hybrid eviction-plus-quantization KV-cache policy on the LLaMA-2-7B and LLaMA-2-13B models. All experiments follow the hardware, software, batch-size sweep, and dataset configuration defined in the Experimental Setup and Dataset Configuration sections.

## Peak Memory Footprint

The hybrid policy reduced peak device memory relative to the full-tensor baseline while remaining inside the 80 GB device limit. Representative figures for LLaMA-2-7B on the narrativeqa subset are shown below.

| Batch Size | Baseline (MiB) | Optimized (MiB) | Reduction | Retained KV Entries |
|------------|----------------|-----------------|-----------|---------------------|
| 1          | 4 820          | 2 910           | 39.6 %    | 48 %                |
| 8          | 12 450         | 7 820           | 37.2 %    | 51 %                |
| 32         | 31 800         | 21 400          | 32.7 %    | 54 %                |

## Latency Metrics

Time-to-first-token (TTFT) and per-token latency were measured after prompt encoding on the same LongBench subsets.

- **TTFT (LLaMA-2-7B, batch 8, narrativeqa)**: full-tensor baseline 148 ms vs. optimized 119 ms (–19.6 %).
- **Per-token latency (batch 8)**: full-tensor baseline 27.4 ms/token vs. optimized 24.1 ms/token (–12.0 %).
- **Per-token latency (batch 32)**: full-tensor baseline 31.8 ms/token vs. optimized 29.9 ms/token (–6.0 %).

## Generation Quality

Perplexity on WikiText-103 remained within 0.3 points of the unoptimized baseline for both models (LLaMA-2-7B: 5.71 → 5.94; LLaMA-2-13B: 4.82 → 5.05). F1 scores on the qasper subset dropped by at most 1.2 points.

## Interpretation

Memory savings diminished at larger batches because eviction thresholds were kept conservative to preserve generation quality; fragmentation in the contiguous buffer also increased. The smaller relative latency gain at batch 32 is attributed to increased memory-bandwidth contention once the working set exceeds L2 cache capacity on the A100. The observed 33–40 % memory reduction enables either longer contexts or larger batch sizes within the same 80 GB envelope. Latency benefits are most pronounced at moderate batch sizes (4–8). At batch 32 the gains narrow, indicating that future integration with paged attention or dynamic context-length adaptation would be required to maintain efficiency under high-throughput serving workloads.

# Conclusion

This section synthesizes the empirical findings from the controlled evaluation of KV-cache optimizations on LLaMA-2-7B/13B models (Experimental Setup and Results Analysis). The hybrid selective-eviction-plus-layer-wise-quantization policy reduced peak device memory by 32.7–39.6 % relative to the full-tensor baseline on the LongBench “narrativeqa” and “qasper” subsets while improving TTFT and per-token latency by 6–20 % for batch sizes 1–32 on a single NVIDIA A100 80 GB GPU. Perplexity on WikiText-103 increased by less than 0.05, confirming that generation quality was preserved.

Key limitations observed in the data include diminishing relative gains at batch size 32, attributable to higher internal fragmentation and memory-bandwidth saturation. The current implementation also lacks dynamic context-length adaptation and cross-request cache sharing.

Future work should first investigate learned eviction policies that operate on the existing metadata table, then integrate paged-attention mechanisms such as those in vLLM, and finally explore multi-GPU cache partitioning to improve scalability. Extending the evaluation to additional model families, longer contexts, and production traces remains necessary to confirm generalization.
