# Reference papers (Markdown)

After `python -m trg search-refs`, download PDFs from the DOIs in `output/references.xlsx`, convert each to `.md`, and save here.

**Naming:** `short_id.md` (e.g. `h2o.md`, `paged_attention.md`)

**Optional title** at the top of the file:

```markdown
---
title: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference"
---

# or use a normal # heading

Paste converted paper text below...
```

Sections **Principles** and **Literature Review** use these files and insert citation blocks:

```markdown
> **[CITATION] Paper Name Here**
> Original: "exact sentence copied from this file"
```
