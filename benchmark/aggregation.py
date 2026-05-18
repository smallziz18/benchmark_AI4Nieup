"""Agrégation et classement des résultats du benchmark."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Tuple

from benchmark.utils import score_global_pct


def rank_models_by_scores(
    summary_rows: List[Dict[str, Any]], weights: Tuple[float, float, float]
) -> List[Dict[str, Any]]:
    """Classe les modèles selon un scénario de pondération donné."""
    ranked = []
    for row in summary_rows:
        ranked.append(
            {
                **row,
                "_scenario_score": score_global_pct(
                    row["score_V"], row["score_E"], row["score_A"], weights
                ),
            }
        )
    ranked.sort(
        key=lambda r: (r["_scenario_score"], r["score_global_pct"], r["score_V"]),
        reverse=True,
    )
    for i, row in enumerate(ranked, start=1):
        row["_scenario_rank"] = i
    return ranked


def robustness_index(ranks: List[int]) -> Tuple[float, float]:
    """Retourne `(std_rang, indice_robustesse)` pour une série de rangs."""
    if len(ranks) < 2:
        return 0.0, 1.0
    std = statistics.pstdev(ranks)
    return round(std, 4), round(1 / (1 + std), 4)
