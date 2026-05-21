# Principles

This section outlines the two theoretical reductions underlying the KV-cache optimizations evaluated in later sections: selective eviction of low-utility entries and layer-wise quantization of retained tensors.

In standard multi-head attention, each newly generated token attends over all preceding key and value projections. The baseline KV cache stores these projections after their first computation, converting the per-token attention cost from quadratic to linear in sequence length while preserving exact numerical equivalence.

> **[CITATION] Literature Review**
> Original: "By retaining the key and value projections of past tokens, it reduces per-token attention complexity from quadratic to linear in sequence length."

Selective eviction is motivated by the observation that attention scores are heavily skewed, with a small subset of tokens (heavy hitters) receiving the majority of attention mass. Maintaining a lightweight importance score derived from cumulative attention or streaming-window statistics permits eviction of entries whose removal produces bounded change to subsequent attention outputs. The policy is applied independently per layer, allowing early layers with more diffuse attention patterns to retain larger caches (default retention 70 % in early layers, 30 % in deeper layers).

Layer-wise quantization exploits variation in the dynamic range of key and value activations across layers. Per-token 4-bit or 8-bit quantization with per-head scale factors is applied selectively to layers whose saliency metrics fall below a chosen threshold. The resulting quantization error is confined to directions of lower impact on attention computations.

Together these reductions lower both the memory footprint and the memory-bandwidth pressure that dominate inference latency on the hardware and batch-size regimes examined in the Experimental Setup section. All latency and memory measurements reported later were obtained on the LongBench benchmark (narrativeqa and qasper subsets) using a single NVIDIA A100 80 GB GPU.
