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
