# Technical Report Generation (TRG)

Automate structured technical reports in **Cursor** using section-based folders, **Grok (xAI)**, and two rule-driven quality passes:

1. **Expert review** — configurable rules in `config/rules.yaml` (`expert_review`)
2. **Fact alignment** — project facts in `config/project.yaml` plus rules (`fact_alignment`)
3. **Question lists** — long, section-grouped questions for gaps you must fill manually
4. **References (M2)** — Crossref search → Excel; Markdown papers → cited Principles / Literature Review

## Layout

```
config/
  project.yaml           # title, topic, ground-truth facts
  rules.yaml             # expert + alignment rules
  reference_search.yaml  # Crossref queries → references.xlsx
sections/                # one folder per report section (edit or LLM-generate)
sources/papers/          # your converted reference .md files
  literature_review/section.md
  problem_analysis/section.md
  ...
output/
  report.md           # assembled report
  references.xlsx     # title, topic, DOI (from search-refs)
  questions.md        # generated questions
  reviews/            # JSON review + alignment artifacts
src/trg/            # Python CLI
```

## Setup

```bash
cd /path/to/Technical_Report_Generation
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`.env` at project root:

```
GROK_API_KEY=xai-...       # from https://console.x.ai
GROK_MODEL=grok-3-mini     # optional, e.g. grok-3, grok-2-1212
```

Edit `config/project.yaml` for your topic and facts. Adjust rules in `config/rules.yaml` and search queries in `config/reference_search.yaml`.

## References workflow (M2)

```bash
# 1) Search Crossref → Excel (Paper title | Topic | DOI)
python -m trg search-refs

# 2) You: download PDFs from DOIs, convert to Markdown → sources/papers/*.md

# 3) Generate sections that quote those files (marked **[CITATION]** blocks)
python -m trg generate -s principles
python -m trg generate -s literature_review
```

See `sources/papers/README.md` for Markdown file format.

## Commands

From project root (with venv active):

```bash
# Draft one section or all
python -m trg generate -s problem_analysis
python -m trg generate

# Expert review (JSON in output/reviews/); --apply writes fixes to section.md
python -m trg review -s implementation
python -m trg review --apply

# Fact alignment vs config facts + cross-section rules
python -m trg align-facts

# Crossref bibliography → output/references.xlsx
python -m trg search-refs

# Long question list for the current project
python -m trg questions --min 50

# Merge sections → output/report.md
python -m trg report

# Full pass: generate → review → align → questions → report
python -m trg pipeline --apply
```

## Cursor workflow

1. Open this repo in Cursor.
2. Edit `config/project.yaml` and section files under `sections/`.
3. Run CLI commands in the integrated terminal, or ask Cursor to run them.
4. Use `output/questions.md` and `output/reviews/*.json` to guide manual fixes; re-run `review` / `align-facts` after edits.

## Example topic

The default config uses *Application and Testing of KV Cache in LLMs* — change `title`, `topic`, and `facts` for your report.
