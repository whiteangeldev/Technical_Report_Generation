from __future__ import annotations

from trg import llm
from trg.config import load_project, project_facts
from trg.paths import OUTPUT_DIR, ensure_output_dirs
from trg.sections import SECTIONS, read_all_sections


def generate_questions(*, min_questions: int = 40) -> str:
    ensure_output_dirs()
    project = load_project()
    facts = project_facts(project)
    sections = read_all_sections()

    outline = "\n".join(
        f"- {title}: {'(has draft)' if (sections.get(sid) or '').strip() else '(empty)'}"
        for sid, title in SECTIONS
    )

    user = f"""Project title: {project.get('title', '')}
Topic: {project.get('topic', '')}
Audience: {project.get('audience', 'technical')}

Known facts:
{chr(10).join(f'- {f}' for f in facts)}

Section status:
{outline}

Generate at least {min_questions} specific questions an expert would ask before
finalizing this technical report. Group by section. Include gaps in literature,
methodology, benchmarks, reproducibility, and limitations.
Format as markdown with ## per section and numbered lists.
"""

    text = llm.complete(
        "You help authors complete rigorous technical reports by asking precise questions.",
        user,
        temperature=0.5,
    )

    path = OUTPUT_DIR / "questions.md"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return str(path)
