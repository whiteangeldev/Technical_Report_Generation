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
