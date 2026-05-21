from __future__ import annotations

from trg import llm
from trg.config import load_project, project_facts
from trg.sections import SECTIONS, read_all_sections, read_section, write_section
from trg.sources import (
    CITATION_FORMAT,
    CITED_SECTIONS,
    format_sources_for_prompt,
    format_sources_index,
    list_papers,
)

SECTION_GUIDANCE: dict[str, str] = {
    "literature_review": "Survey prior work and cite representative approaches.",
    "problem_analysis": "Define the problem, constraints, and why existing methods fall short.",
    "new_technology": "Describe the adopted technology/solution at a high level.",
    "principles": "Explain underlying principles and theory of the new technology.",
    "implementation": "Describe implementation steps, architecture, and key design choices.",
    "experimental_setup": "Hardware, software stack, baselines, metrics, and procedure.",
    "dataset_configuration": "Datasets, splits, preprocessing, and evaluation protocol.",
    "results_analysis": "Present and interpret results with tables or bullet metrics.",
    "conclusion": "Summarize findings, limitations, and future work.",
}


def generate_section(section_id: str) -> str:
    project = load_project()
    facts = project_facts(project)
    title = next(t for sid, t in SECTIONS if sid == section_id)
    guidance = SECTION_GUIDANCE[section_id]
    existing = read_section(section_id).strip()
    others = read_all_sections()

    context_parts = []
    for sid, stitle in SECTIONS:
        if sid == section_id:
            continue
        body = (others.get(sid) or "").strip()
        if body:
            context_parts.append(f"### {stitle}\n{body[:2000]}")

    citation_block = ""
    if section_id in CITED_SECTIONS:
        papers = list_papers()
        sources_text = format_sources_for_prompt(papers)
        citation_block = f"""
Citation rules (required for this section):
{CITATION_FORMAT}

Available sources:
{format_sources_index(papers)}

Reference markdown to quote from:
{sources_text or '(add .md files under sources/papers/ — see sources/papers/README.md)'}
"""

    user = f"""Write the "{title}" section for this technical report.

Project: {project.get('title', '')}
Topic: {project.get('topic', '')}
Language: {project.get('language', 'en')}

Section focus: {guidance}

Facts to respect:
{chr(10).join(f'- {f}' for f in facts)}
{citation_block}
Other sections for consistency (do not repeat their full content):
{chr(10).join(context_parts) if context_parts else '(none yet)'}

Current draft (improve or replace if present):
---
{existing or '(empty)'}
---

Output markdown only. Start with # {title}. Be concrete; use placeholders like [TBD]
only where the user must supply real data.
"""

    system = (
        "You write clear, structured technical report sections for engineering audiences."
    )
    if section_id in CITED_SECTIONS:
        system += (
            " When reference markdown is provided, mark every direct quote with the "
            "**[CITATION]** block format exactly as specified."
        )

    content = llm.complete(system, user)
    write_section(section_id, content)
    return content
