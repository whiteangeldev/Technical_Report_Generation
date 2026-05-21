from __future__ import annotations

from typing import Any

import yaml

from trg.paths import PROJECT_FILE, RULES_FILE


def _load_yaml(path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_project() -> dict[str, Any]:
    return _load_yaml(PROJECT_FILE)


def load_rules() -> dict[str, list[str]]:
    data = _load_yaml(RULES_FILE)
    return {
        "expert_review": data.get("expert_review") or [],
        "fact_alignment": data.get("fact_alignment") or [],
    }


def project_facts(project: dict[str, Any] | None = None) -> list[str]:
    proj = project or load_project()
    facts = proj.get("facts") or []
    return [str(f).strip() for f in facts if str(f).strip()]
