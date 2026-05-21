# New Technology

This section outlines the hybrid KV-cache optimization whose performance is measured in the Results Analysis section.

The technique integrates H2O-style heavy-hitter eviction with a sliding-window streaming policy to discard low-attention-score entries, followed by per-layer 4-bit or 8-bit quantization applied only to the retained tensors of lower-importance layers. Importance scores are maintained in a lightweight metadata table and updated after each decoding step, enabling eviction decisions without host synchronization. The combined policy avoids both full-tensor retention and operating-system-level paging while preserving the original attention computation for kept entries.
