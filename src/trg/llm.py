from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from trg.paths import ROOT

load_dotenv(ROOT / ".env")

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = os.getenv("GROK_MODEL") or os.getenv("XAI_MODEL") or "grok-3-mini"


def _api_key() -> str:
    key = (os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or "").strip()
    if key.startswith("GROK_API_KEY=") or key.startswith("XAI_API_KEY="):
        key = key.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError(
            "GROK_API_KEY missing. Set it in .env at project root (get one at https://console.x.ai)."
        )
    return key


def client() -> OpenAI:
    base_url = (os.getenv("GROK_BASE_URL") or os.getenv("XAI_BASE_URL") or XAI_BASE_URL).strip()
    return OpenAI(api_key=_api_key(), base_url=base_url)


def complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.4,
) -> str:
    response = client().chat.completions.create(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def complete_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
) -> dict:
    text = complete(
        system + "\nRespond with valid JSON only, no markdown fences.",
        user,
        model=model,
        temperature=0.2,
    )
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return json.loads(text)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
