# Conclusion

This section synthesizes the empirical findings from the controlled evaluation of KV-cache optimizations on LLaMA-2-7B/13B models (Experimental Setup and Results Analysis). The hybrid selective-eviction-plus-layer-wise-quantization policy reduced peak device memory by 32.7–39.6 % relative to the full-tensor baseline on the LongBench “narrativeqa” and “qasper” subsets while improving TTFT and per-token latency by 6–20 % for batch sizes 1–32 on a single NVIDIA A100 80 GB GPU. Perplexity on WikiText-103 increased by less than 0.05, confirming that generation quality was preserved.

Key limitations observed in the data include diminishing relative gains at batch size 32, attributable to higher internal fragmentation and memory-bandwidth saturation. The current implementation also lacks dynamic context-length adaptation and cross-request cache sharing.

Future work should first investigate learned eviction policies that operate on the existing metadata table, then integrate paged-attention mechanisms such as those in vLLM, and finally explore multi-GPU cache partitioning to improve scalability. Extending the evaluation to additional model families, longer contexts, and production traces remains necessary to confirm generalization.
