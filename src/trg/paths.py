from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
SECTIONS_DIR = ROOT / "sections"
OUTPUT_DIR = ROOT / "output"
REVIEWS_DIR = OUTPUT_DIR / "reviews"

PROJECT_FILE = CONFIG_DIR / "project.yaml"
RULES_FILE = CONFIG_DIR / "rules.yaml"
REFERENCE_SEARCH_FILE = CONFIG_DIR / "reference_search.yaml"

SOURCES_DIR = ROOT / "sources"
PAPERS_DIR = SOURCES_DIR / "papers"
REFERENCES_XLSX = OUTPUT_DIR / "references.xlsx"


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_sources_dirs() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
