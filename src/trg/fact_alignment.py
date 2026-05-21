from __future__ import annotations

from datetime import datetime, timezone

from trg import llm
from trg.config import load_project, load_rules, project_facts
from trg.paths import REVIEWS_DIR, ensure_output_dirs
from trg.sections import read_all_sections


def align_facts(section_id: str | None = None) -> dict:
    """Check report content against config facts and cross-section consistency."""
    ensure_output_dirs()
    project = load_project()
    rules = load_rules()["fact_alignment"]
    facts = project_facts(project)

    if section_id:
        from trg.sections import read_section

        sections = {section_id: read_section(section_id)}
    else:
        sections = read_all_sections()

    combined = "\n\n".join(
        f"## {sid}\n{(text or '').strip()}" for sid, text in sections.items() if (text or "").strip()
    )

    user = f"""Project: {project.get('title', '')}
Topic: {project.get('topic', '')}

Configured facts (must hold):
{chr(10).join(f'- {f}' for f in facts)}

Fact-alignment rules:
{chr(10).join(f'- {r}' for r in rules)}

Report content:
---
{combined or '(no section content yet)'}
---
"""

    result = llm.complete_json(
        "You are a fact-checking editor for technical reports. Return JSON with keys: "
        "aligned (boolean), violations (array of {{fact, issue, section}}), "
        "cross_section_issues (string[]), recommendations (string[]).",
        user,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = section_id or "all"
    out_path = REVIEWS_DIR / f"facts_{label}_{stamp}.json"
    llm.save_json(out_path, result)
    result["review_file"] = str(out_path)
    return result


def write_alignment_summary(results: list[dict] | None = None) -> str:
    from trg.paths import OUTPUT_DIR

    ensure_output_dirs()
    if results is None:
        results = [align_facts()]

    lines = ["# Fact alignment summary\n"]
    for r in results:
        lines.append(f"- **Aligned:** {r.get('aligned', 'unknown')}")
        for v in r.get("violations") or []:
            lines.append(f"  - Violation ({v.get('section', '?')}): {v.get('issue', '')}")
        for issue in r.get("cross_section_issues") or []:
            lines.append(f"  - Cross-section: {issue}")
        lines.append(f"  - Detail: `{r.get('review_file', '')}`\n")

    path = REVIEWS_DIR / "fact_alignment_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
