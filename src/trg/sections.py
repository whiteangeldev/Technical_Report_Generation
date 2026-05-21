from pathlib import Path

from trg.paths import SECTIONS_DIR

SECTIONS: list[tuple[str, str]] = [
    ("literature_review", "Literature Review"),
    ("problem_analysis", "Problem Analysis"),
    ("new_technology", "New Technology"),
    ("principles", "Principles"),
    ("implementation", "Implementation"),
    ("experimental_setup", "Experimental Setup"),
    ("dataset_configuration", "Dataset Configuration"),
    ("results_analysis", "Results Analysis"),
    ("conclusion", "Conclusion"),
]

SECTION_IDS = [s[0] for s in SECTIONS]


def section_path(section_id: str) -> Path:
    if section_id not in SECTION_IDS:
        raise ValueError(f"Unknown section: {section_id}. Choose from: {', '.join(SECTION_IDS)}")
    return SECTIONS_DIR / section_id / "section.md"


def read_section(section_id: str) -> str:
    path = section_path(section_id)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_section(section_id: str, content: str) -> Path:
    path = section_path(section_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def read_all_sections() -> dict[str, str]:
    return {sid: read_section(sid) for sid in SECTION_IDS}


def assemble_report() -> str:
    parts: list[str] = []
    for sid, title in SECTIONS:
        body = read_section(sid).strip()
        if body:
            parts.append(body)
        else:
            parts.append(f"# {title}\n\n_(empty — run generate for this section)_\n")
    return "\n\n".join(parts) + "\n"
