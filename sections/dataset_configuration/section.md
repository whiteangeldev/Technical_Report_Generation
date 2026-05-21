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
