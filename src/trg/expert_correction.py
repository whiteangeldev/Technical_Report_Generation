from __future__ import annotations

from datetime import datetime, timezone

from trg import llm
from trg.config import load_project, load_rules
from trg.paths import REVIEWS_DIR, ensure_output_dirs
from trg.sections import SECTIONS, read_section, write_section


def _context_block(other_sections: dict[str, str], current_id: str) -> str:
    lines: list[str] = []
    for sid, title in SECTIONS:
        if sid == current_id:
            continue
        text = (other_sections.get(sid) or "").strip()
        if not text or len(text) < 80:
            continue
        preview = text[:1200] + ("..." if len(text) > 1200 else "")
        lines.append(f"### {title}\n{preview}")
    return "\n\n".join(lines) if lines else "(no other section drafts yet)"


def review_section(section_id: str, *, apply_fix: bool = False) -> dict:
    ensure_output_dirs()
    project = load_project()
    rules = load_rules()["expert_review"]
    from trg.sections import read_all_sections

    content = read_section(section_id)
    title = next(t for sid, t in SECTIONS if sid == section_id)
    others = read_all_sections()

    user = f"""Project: {project.get('title', '')}
Topic: {project.get('topic', '')}

Section to review ({title}):
---
{content or '(empty)'}
---

Other sections (for consistency, do not rewrite them):
---
{_context_block(others, section_id)}
---

Expert review rules:
{chr(10).join(f'- {r}' for r in rules)}
"""

    result = llm.complete_json(
        "You are a senior technical report reviewer. Review the section from an expert "
        "perspective. Return JSON with keys: score (1-10), issues (string[]), "
        "suggestions (string[]), corrected_markdown (full improved section as markdown).",
        user,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = REVIEWS_DIR / f"{section_id}_{stamp}.json"
    llm.save_json(out_path, result)

    if apply_fix and result.get("corrected_markdown"):
        write_section(section_id, result["corrected_markdown"])

    result["review_file"] = str(out_path)
    return result


def review_all(*, apply_fix: bool = False) -> list[dict]:
    from trg.sections import SECTION_IDS

    return [review_section(sid, apply_fix=apply_fix) for sid in SECTION_IDS]
