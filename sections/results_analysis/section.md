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
