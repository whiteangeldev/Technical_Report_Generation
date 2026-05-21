from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from openpyxl import Workbook

from trg.config import load_reference_search
from trg.paths import REFERENCES_XLSX, ensure_output_dirs


def _crossref_search(query: str, rows: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({"query": query, "rows": rows})
    url = f"https://api.crossref.org/works?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "trg/0.1 (mailto:local)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    items = data.get("message", {}).get("items") or []
    results: list[dict[str, str]] = []
    for item in items:
        titles = item.get("title") or []
        title = (titles[0] if titles else "").strip()
        doi = (item.get("DOI") or "").strip()
        if title and doi:
            results.append({"title": title, "doi": doi})
    return results


def search_references() -> list[dict[str, str]]:
    """Run configured Crossref queries; return rows with title, topic, doi."""
    cfg = load_reference_search()
    max_per = int(cfg.get("max_per_query") or 5)
    searches = cfg.get("searches") or []
    if not searches:
        raise RuntimeError(
            "No searches in config/reference_search.yaml. Add at least one query/topic pair."
        )

    seen_dois: set[str] = set()
    rows: list[dict[str, str]] = []
    for entry in searches:
        query = str(entry.get("query") or "").strip()
        topic = str(entry.get("topic") or "General").strip()
        if not query:
            continue
        for hit in _crossref_search(query, max_per):
            doi = hit["doi"].lower()
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
            rows.append({"title": hit["title"], "topic": topic, "doi": hit["doi"]})
    return rows


def write_references_xlsx(rows: list[dict[str, str]], path=REFERENCES_XLSX) -> str:
    ensure_output_dirs()
    wb = Workbook()
    ws = wb.active
    ws.title = "references"
    ws.append(["Paper title", "Topic", "DOI"])
    for row in rows:
        ws.append([row["title"], row["topic"], row["doi"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return str(path)


def run_search_refs() -> str:
    rows = search_references()
    if not rows:
        raise RuntimeError("No references found. Check queries in config/reference_search.yaml.")
    out = write_references_xlsx(rows)
    return out
