from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trg.paths import PAPERS_DIR, ensure_sources_dirs

# Sections that pull quotes from sources/papers/*.md
CITED_SECTIONS = frozenset({"principles", "literature_review"})

MAX_CHARS_PER_SOURCE = 12_000

CITATION_FORMAT = """\
When you quote or closely paraphrase a source, use this block (verbatim excerpt only):

> **[CITATION] {paper_name}**
> Original: "{exact words from the source markdown below}"

- Put your own analysis outside these blocks.
- Do not invent quotes; only use text that appears in the sources.
- Use at least one citation block when sources are provided.
"""


@dataclass
class SourceDoc:
    paper_id: str
    paper_name: str
    path: Path
    body: str


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip().lower()] = val.strip().strip('"').strip("'")
    return meta, body


def _paper_name_from_body(body: str, fallback: str) -> str:
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_paper(path: Path) -> SourceDoc:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_front_matter(raw)
    paper_id = path.stem
    name = meta.get("title") or _paper_name_from_body(body, paper_id.replace("_", " "))
    return SourceDoc(paper_id=paper_id, paper_name=name, path=path, body=body.strip())


def list_papers() -> list[SourceDoc]:
    ensure_sources_dirs()
    docs: list[SourceDoc] = []
    for path in sorted(PAPERS_DIR.glob("*.md")):
        if path.name.startswith("_"):
            continue
        docs.append(load_paper(path))
    return docs


def format_sources_for_prompt(docs: list[SourceDoc]) -> str:
    if not docs:
        return ""
    parts: list[str] = []
    for doc in docs:
        body = doc.body
        if len(body) > MAX_CHARS_PER_SOURCE:
            body = body[:MAX_CHARS_PER_SOURCE] + "\n\n[... truncated ...]"
        parts.append(f"### Source: {doc.paper_name} (file: {doc.path.name})\n{body}")
    return "\n\n".join(parts)


def format_sources_index(docs: list[SourceDoc]) -> str:
    if not docs:
        return "(no papers in sources/papers/ yet)"
    return "\n".join(f"- {d.paper_name} (`{d.path.name}`)" for d in docs)
