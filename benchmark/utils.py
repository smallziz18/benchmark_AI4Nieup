"""Utilitaires partagés du benchmark."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Sequence, Tuple

from benchmark.catalog import DEFAULT_WEIGHTS


def strip_code_fences(text: str) -> str:
    """Retire les fences markdown et renvoie la payload texte utile."""
    cleaned = text.strip()
    if "```" not in cleaned:
        return cleaned
    for part in cleaned.split("```"):
        candidate = part.strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    return cleaned


def parse_json_payload(text: str) -> Dict[str, Any]:
    """Parse un JSON possiblement entouré de texte/fences markdown."""
    raw = strip_code_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Convertit une valeur en int de façon robuste, sinon `default`."""
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            return int(round(float(value)))
        return int(round(float(str(value).strip())))
    except Exception:
        return default


def clamp_score(value: Any) -> Optional[int]:
    """Normalise un score sur l'intervalle [0, 100]."""
    score = safe_int(value, None)
    if score is None:
        return None
    return max(0, min(100, score))


def score_global_pct(
    score_v: int,
    score_e: int,
    score_a: int,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> float:
    """Calcule le score global pondéré en pourcentage."""
    wv, we, wa = weights
    return round(score_v * wv + score_e * we + score_a * wa, 2)


def mean_or_zero(values: Sequence[Optional[float]]) -> float:
    """Moyenne sur valeurs numériques valides, sinon 0.0."""
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def std_or_zero(values: Sequence[float]) -> float:
    """Écart-type population, ou 0.0 avec moins de 2 valeurs."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return round(math.sqrt(variance), 4)
