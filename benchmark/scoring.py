"""Juge LLM et calcul des scores pédagogiques."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from benchmark.catalog import DEFAULT_WEIGHTS, JUDGE_MODELS
from benchmark.mlflow_utils import (
    MLflowConfig,
    log_artifact_path,
    log_dataframe,
    log_metrics,
    log_params,
    start_mlflow_run,
)
from benchmark.prompts import JUDGE_PROMPT, JUDGE_SYSTEM
from benchmark.providers import (
    detect_provider,
    estimate_cost_usd,
    get_openai_client,
    list_ollama_models,
)
from benchmark.utils import (
    clamp_score,
    mean_or_zero,
    parse_json_payload,
    score_global_pct,
)


def _chat_completion(
    client: OpenAI,
    provider: str,
    model_id: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_k: Optional[int] = None,
) -> Tuple[str, Optional[int], Optional[int]]:
    """Envoie une completion au juge et retourne texte + tokens usage."""
    extra_headers = None
    if provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": "https://ia-pour-tous.sn",
            "X-Title": "IA Pour Tous Judge",
        }
    kwargs: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_headers": extra_headers,
    }
    # top_k n'est pas supporté nativement par l'API OpenAI directe.
    if top_k is not None and provider in {"openrouter", "ollama"}:
        kwargs["extra_body"] = {"top_k": int(top_k)}
    resp = client.chat.completions.create(**kwargs)
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", None) if usage else None
    out_tok = getattr(usage, "completion_tokens", None) if usage else None
    return resp.choices[0].message.content or "", in_tok, out_tok


def judge_one(
    client: OpenAI,
    provider: str,
    model_id: str,
    row: Dict[str, Any],
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
    temperature: float = 0.0,
    max_tokens: int = 600,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Évalue une réponse unique et retourne les scores V/E/A + flags."""
    prompt = JUDGE_PROMPT.format(
        competence=row.get("competence", ""),
        niveau=row.get("niveau", ""),
        sujet=row.get("sujet", ""),
        question=row.get("question", ""),
        reponse_ideale=row.get("reponse_ideale", ""),
        critere=row.get("critere", ""),
        response=row.get("response", ""),
    )
    raw, in_tok, out_tok = _chat_completion(
        client,
        provider,
        model_id,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=top_k,
    )

    try:
        parsed = parse_json_payload(raw)
    except Exception as exc:
        parsed = {
            "score_V": None,
            "score_E": None,
            "score_A": None,
            "points_forts": "",
            "points_faibles": "",
            "justification": f"Erreur parsing JSON: {exc}. Raw: {raw[:200]}",
            "adapte_niveau": None,
        }

    score_v = clamp_score(parsed.get("score_V"))
    score_e = clamp_score(parsed.get("score_E"))
    score_a = clamp_score(parsed.get("score_A"))
    if score_v is None and clamp_score(parsed.get("score")) is not None:
        legacy = clamp_score(parsed.get("score")) or 0
        score_v = score_e = score_a = max(0, min(100, legacy * 10))
    score_v = 0 if score_v is None else score_v
    score_e = 0 if score_e is None else score_e
    score_a = 0 if score_a is None else score_a

    score_pct = score_global_pct(score_v, score_e, score_a, weights)
    return {
        "score_V": score_v,
        "score_E": score_e,
        "score_A": score_a,
        "score_global_pct": score_pct,
        "score": round(score_pct / 10.0, 2),
        "points_forts": parsed.get("points_forts", "") or "",
        "points_faibles": parsed.get("points_faibles", "") or "",
        "justification": parsed.get("justification", "") or "",
        "adapte_niveau": parsed.get("adapte_niveau", None),
        "flag_hallucination": score_v < 50,
        "flag_non_etayage": score_e < 40,
        "flag_desadaptation": score_a < 30,
        "provider": provider,
        "judge_model_id": model_id,
        "judge_input_tokens": in_tok,
        "judge_output_tokens": out_tok,
        "judge_call_cost_usd": estimate_cost_usd(provider, model_id, in_tok, out_tok),
    }


def _judge_one_response(
    row_index: int,
    row_dict: Dict[str, Any],
    judge_clients: Dict[str, Tuple[str, str, Optional[int], OpenAI]],
    weights: Tuple[float, float, float],
) -> Tuple[int, List[Dict[str, Any]]]:
    """Scorer une réponse avec tous les juges. Retourne (index, per_judge_results)."""
    per_judge_results = []
    for judge_key, (provider, model_id, top_k, client) in judge_clients.items():
        try:
            per_judge_results.append(
                {
                    "judge": judge_key,
                    **judge_one(
                        client,
                        provider,
                        model_id,
                        row_dict,
                        weights=weights,
                        top_k=top_k,
                    ),
                }
            )
        except Exception as exc:
            per_judge_results.append(
                {
                    "judge": judge_key,
                    "score_V": 0,
                    "score_E": 0,
                    "score_A": 0,
                    "score_global_pct": 0.0,
                    "score": 0.0,
                    "points_forts": "",
                    "points_faibles": "",
                    "justification": f"Erreur du juge: {exc}",
                    "adapte_niveau": None,
                    "flag_hallucination": True,
                    "flag_non_etayage": True,
                    "flag_desadaptation": True,
                    "provider": provider,
                    "judge_model_id": model_id,
                },
            )
    return row_index, per_judge_results


def _summary_rows(
    scored: pd.DataFrame, judge_keys: Sequence[str], timestamp: str
) -> pd.DataFrame:
    """Construit les lignes du résumé agrégé par modèle."""

    def _q(series: pd.Series, q: float, ndigits: int = 3):
        vals = series.dropna()
        if vals.empty:
            return None
        return round(float(vals.quantile(q)), ndigits)

    rows: List[Dict[str, Any]] = []
    for model_name, grp in scored.groupby("model_name"):
        lat = (
            grp["latency_s"].dropna()
            if "latency_s" in grp.columns
            else pd.Series(dtype=float)
        )
        ttft = (
            grp["ttft_s"].dropna()
            if "ttft_s" in grp.columns
            else pd.Series(dtype=float)
        )
        tpot = (
            grp["tpot_s"].dropna()
            if "tpot_s" in grp.columns
            else pd.Series(dtype=float)
        )
        in_tok = (
            grp["input_tokens"].dropna()
            if "input_tokens" in grp.columns
            else pd.Series(dtype=float)
        )
        out_tok = (
            grp["output_tokens"].dropna()
            if "output_tokens" in grp.columns
            else pd.Series(dtype=float)
        )

        total_latency = float(lat.sum()) if not lat.empty else 0.0
        total_out_tokens = float(out_tok.sum()) if not out_tok.empty else 0.0
        total_in_tokens = float(in_tok.sum()) if not in_tok.empty else 0.0
        debit_tokens_s = (
            round(total_out_tokens / total_latency, 3) if total_latency > 0 else None
        )

        row_rates = []
        if "output_tokens" in grp.columns and "latency_s" in grp.columns:
            for _, r in grp[["output_tokens", "latency_s"]].dropna().iterrows():
                latency_val = float(r["latency_s"])
                if latency_val > 0:
                    row_rates.append(float(r["output_tokens"]) / latency_val)

        lat_moy = round(float(lat.mean()), 2) if not lat.empty else None
        req_min = round(60.0 / lat_moy, 2) if lat_moy and lat_moy > 0 else None

        generation_cost_total = (
            round(float(grp["estimated_cost_usd"].dropna().sum()), 6)
            if "estimated_cost_usd" in grp.columns
            else 0.0
        )
        judge_cost_total = (
            round(float(grp["judge_cost_usd"].dropna().sum()), 6)
            if "judge_cost_usd" in grp.columns
            else 0.0
        )
        total_cost = round(generation_cost_total + judge_cost_total, 6)
        nb_questions = int(len(grp))

        rows.append(
            {
                "rang": 0,
                "model": model_name,
                "tier": grp["tier"].iloc[0] if "tier" in grp.columns else "unknown",
                "score_V": round(grp["score_V"].mean(), 2),
                "score_E": round(grp["score_E"].mean(), 2),
                "score_A": round(grp["score_A"].mean(), 2),
                "score_global_pct": round(grp["score_global_pct"].mean(), 2),
                "score_global": round(grp["score"].mean(), 2),
                "success_V": round((grp["score_V"] >= 70).mean() * 100, 2),
                "success_E": round((grp["score_E"] >= 70).mean() * 100, 2),
                "success_A": round((grp["score_A"] >= 70).mean() * 100, 2),
                "hallucination_rate": round((grp["score_V"] < 50).mean() * 100, 2),
                "non_etayage_rate": round((grp["score_E"] < 40).mean() * 100, 2),
                "desadaptation_rate": round((grp["score_A"] < 30).mean() * 100, 2),
                "latence_moy_s": lat_moy,
                "latence_med_s": _q(lat, 0.5, 3),
                "latence_p90_s": _q(lat, 0.9, 3),
                "latence_p95_s": _q(lat, 0.95, 3),
                "ttft_moy_s": round(float(ttft.mean()), 3) if not ttft.empty else None,
                "ttft_med_s": _q(ttft, 0.5, 3),
                "ttft_p90_s": _q(ttft, 0.9, 3),
                "tpot_moy_s": round(float(tpot.mean()), 4) if not tpot.empty else None,
                "tpot_med_s": _q(tpot, 0.5, 4),
                "input_tokens_moy": (
                    round(grp["input_tokens"].dropna().mean(), 0)
                    if "input_tokens" in grp.columns
                    and grp["input_tokens"].notna().any()
                    else None
                ),
                "output_tokens_moy": (
                    round(grp["output_tokens"].dropna().mean(), 0)
                    if "output_tokens" in grp.columns
                    and grp["output_tokens"].notna().any()
                    else None
                ),
                "input_tokens_total": round(total_in_tokens, 0),
                "output_tokens_total": round(total_out_tokens, 0),
                "debit_tokens_s_moy": debit_tokens_s,
                "debit_tokens_s_med": (
                    round(float(pd.Series(row_rates).median()), 3)
                    if row_rates
                    else None
                ),
                "requetes_par_min_estime": req_min,
                "nb_questions": nb_questions,
                "nb_erreurs": (
                    int(grp["error"].notna().sum()) if "error" in grp.columns else 0
                ),
                "generation_cost_total_usd": generation_cost_total,
                "judge_cost_total_usd": judge_cost_total,
                "total_cost_usd": total_cost,
                "cost_per_question_usd": (
                    round(total_cost / nb_questions, 6) if nb_questions else None
                ),
                "juge": ",".join(judge_keys),
                "timestamp": timestamp,
            }
        )
    return pd.DataFrame(rows)


def run_judge(
    input_path: str,
    judge_keys: Sequence[str],
    output_dir: Optional[str] = None,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
    workers: Optional[int] = None,
    mlflow_config: MLflowConfig | None = None,
) -> Tuple[str, str]:
    """Exécute la notation multi-juges et écrit `scored_*.csv` + `summary_*.csv`."""
    input_file = Path(input_path)
    if not input_file.exists():
        fallback = Path("benchmark_results") / input_file.name
        if fallback.exists():
            input_file = fallback
    input_path = str(input_file)
    df = pd.read_csv(input_path)
    if output_dir is None:
        output_dir = str(Path(input_path).resolve().parent)
    os.makedirs(output_dir, exist_ok=True)

    if "error" not in df.columns:
        df["error"] = None

    valid_mask = (
        df["error"].isna()
        & df["response"].notna()
        & (df["response"].astype(str).str.strip() != "")
    )
    valid = df[valid_mask].copy()
    skip = df[~valid_mask].copy()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = (
        mlflow_config.run_name
        if mlflow_config and mlflow_config.run_name
        else f"judge:{Path(input_path).stem}:{timestamp}"
    )

    print("\nLLM Judge")
    print(f"Fichier d'entree : {os.path.basename(input_path)}")
    print(f"Juges            : {', '.join(judge_keys)}")
    print(f"Reponses a noter : {len(valid)}")

    with start_mlflow_run(mlflow_config, nested=False, run_name=run_name) as mlflow:
        if mlflow is not None:
            log_params(
                mlflow,
                {
                    "stage": "judge",
                    "input_path": input_path,
                    "input_name": Path(input_path).name,
                    "output_dir": output_dir,
                    "judge_keys": ",".join(judge_keys),
                    "weights_V": weights[0],
                    "weights_E": weights[1],
                    "weights_A": weights[2],
                    "nb_questions_to_note": len(valid),
                    "nb_questions_ignored": len(skip),
                },
            )
            log_metrics(
                mlflow,
                {"nb_questions_to_note": len(valid), "nb_questions_ignored": len(skip)},
            )

        local_models = list_ollama_models()
        judge_clients: Dict[str, Tuple[str, str, Optional[int], OpenAI]] = {}
        for judge_key in judge_keys:
            if judge_key in JUDGE_MODELS:
                cfg = JUDGE_MODELS[judge_key]
                provider = cfg["provider"]
                model_id = cfg["model_id"]
                top_k = int(cfg["top_k"]) if cfg.get("top_k") is not None else None
            else:
                _, provider, model_id, cfg = detect_provider(judge_key, local_models)
                top_k = int(cfg["top_k"]) if cfg.get("top_k") is not None else None
            judge_clients[judge_key] = (
                provider,
                model_id,
                top_k,
                get_openai_client(provider),
            )

        scored_rows: List[Dict[str, Any]] = []

        # Paralléliser les appels de notation avec ThreadPoolExecutor
        # La valeur par défaut est conservatrice, mais reste configurable via CLI.
        max_workers = max(
            1, min(int(workers or 4), len(judge_clients), len(valid) or 1)
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for row_index, (_, row) in enumerate(valid.iterrows()):
                row_dict = row.to_dict()
                future = executor.submit(
                    _judge_one_response, row_index, row_dict, judge_clients, weights
                )
                futures[future] = row_index

            # Traiter les résultats au fur et à mesure
            with tqdm(total=len(valid), ncols=78, desc="  Notation") as pbar:
                for future in as_completed(futures):
                    row_index, per_judge_results = future.result()
                    # Récupérer la ligne originale
                    row = valid.iloc[row_index]
                    row_dict = row.to_dict()

                    avg_v = mean_or_zero([r["score_V"] for r in per_judge_results])
                    avg_e = mean_or_zero([r["score_E"] for r in per_judge_results])
                    avg_a = mean_or_zero([r["score_A"] for r in per_judge_results])
                    global_pct = round(
                        avg_v * weights[0] + avg_e * weights[1] + avg_a * weights[2], 2
                    )
                    judge_input_tokens = int(
                        sum(
                            int(r.get("judge_input_tokens") or 0)
                            for r in per_judge_results
                        )
                    )
                    judge_output_tokens = int(
                        sum(
                            int(r.get("judge_output_tokens") or 0)
                            for r in per_judge_results
                        )
                    )
                    judge_cost_usd = round(
                        sum(
                            float(r.get("judge_call_cost_usd") or 0.0)
                            for r in per_judge_results
                        ),
                        6,
                    )

                    scored_rows.append(
                        {
                            **row_dict,
                            "score_V": avg_v,
                            "score_E": avg_e,
                            "score_A": avg_a,
                            "score_global_pct": global_pct,
                            "score": round(global_pct / 10.0, 2),
                            "flag_hallucination": avg_v < 50,
                            "flag_non_etayage": avg_e < 40,
                            "flag_desadaptation": avg_a < 30,
                            "judge": ",".join(judge_keys),
                            "judge_input_tokens": judge_input_tokens,
                            "judge_output_tokens": judge_output_tokens,
                            "judge_cost_usd": judge_cost_usd,
                            "judge_details": json.dumps(
                                {
                                    "weights": {
                                        "V": weights[0],
                                        "E": weights[1],
                                        "A": weights[2],
                                    },
                                    "judges": per_judge_results,
                                },
                                ensure_ascii=False,
                            ),
                            "score_details": json.dumps(
                                {
                                    "score_V": avg_v,
                                    "score_E": avg_e,
                                    "score_A": avg_a,
                                    "score_global_pct": global_pct,
                                    "judges": per_judge_results,
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    pbar.update(1)

        skip = skip.copy()
        skip["score_V"] = 0
        skip["score_E"] = 0
        skip["score_A"] = 0
        skip["score_global_pct"] = 0
        skip["score"] = -1
        skip["flag_hallucination"] = False
        skip["flag_non_etayage"] = False
        skip["flag_desadaptation"] = False
        skip["judge"] = ",".join(judge_keys)
        skip["judge_input_tokens"] = 0
        skip["judge_output_tokens"] = 0
        skip["judge_cost_usd"] = 0.0
        skip["judge_details"] = json.dumps(
            {"ignored": True, "reason": "Reponse vide ou erreur d'appel API"},
            ensure_ascii=False,
        )
        skip["score_details"] = json.dumps(
            {
                "score": -1,
                "justification": "Ignore - reponse vide ou erreur d'appel API",
            },
            ensure_ascii=False,
        )

        final = (
            pd.concat([pd.DataFrame(scored_rows), skip], ignore_index=True)
            .sort_values("id")
            .reset_index(drop=True)
        )
        out_path = str(Path(output_dir) / f"scored_{timestamp}.csv")
        final.to_csv(out_path, index=False, encoding="utf-8")

        scored = final[final["score"] >= 0].copy()
        summary = (
            _summary_rows(scored, judge_keys, timestamp)
            if not scored.empty
            else pd.DataFrame(columns=["rang", "model", "tier"])
        )
        if not summary.empty:
            summary = summary.sort_values(
                ["score_global_pct", "score_V", "score_E"], ascending=False
            ).reset_index(drop=True)
            summary["rang"] = range(1, len(summary) + 1)
            best_score = float(summary["score_global_pct"].max())
            best_speed = (
                float(summary["latence_moy_s"].dropna().min())
                if summary["latence_moy_s"].notna().any()
                else None
            )
            summary["vs_best_delta_score"] = (
                summary["score_global_pct"] - best_score
            ).round(2)
            summary["vs_best_delta_latence_s"] = (
                (summary["latence_moy_s"] - best_speed).round(3)
                if best_speed is not None
                else None
            )
            summary["quality_speed_index"] = (
                summary["score_global_pct"] / summary["latence_moy_s"].replace(0, pd.NA)
            ).round(3)
            summary["rank_qualite"] = (
                summary["score_global_pct"]
                .rank(method="min", ascending=False)
                .astype("Int64")
            )
            summary["rank_vitesse"] = (
                summary["latence_moy_s"]
                .rank(method="min", ascending=True)
                .astype("Int64")
            )
            summary["rank_debit"] = (
                summary["debit_tokens_s_moy"]
                .rank(method="min", ascending=False, na_option="bottom")
                .astype("Int64")
            )
            summary["rank_composite_qv"] = (
                summary["rank_qualite"] * 0.6 + summary["rank_vitesse"] * 0.4
            ).round(2)

        summary_path = str(Path(output_dir) / f"summary_{timestamp}.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8")

        if mlflow is not None:
            log_artifact_path(mlflow, out_path, artifact_path="benchmark_outputs")
            log_artifact_path(mlflow, summary_path, artifact_path="benchmark_outputs")
            log_dataframe(
                mlflow,
                final,
                f"scored_{timestamp}.csv",
                artifact_path="benchmark_outputs/full_scored",
            )
            if not summary.empty:
                best_row = summary.iloc[0]
                log_metrics(
                    mlflow,
                    {
                        "nb_models": float(len(summary)),
                        "nb_scored_questions": float(len(scored)),
                        "best_score_global_pct": float(best_row["score_global_pct"]),
                        "best_score_V": float(best_row["score_V"]),
                        "best_score_E": float(best_row["score_E"]),
                        "best_score_A": float(best_row["score_A"]),
                        "best_latence_moy_s": (
                            float(best_row["latence_moy_s"])
                            if pd.notna(best_row["latence_moy_s"])
                            else None
                        ),
                        "best_cost_per_question_usd": (
                            float(best_row["cost_per_question_usd"])
                            if pd.notna(best_row["cost_per_question_usd"])
                            else None
                        ),
                        "judge_cost_total_usd": float(
                            summary["judge_cost_total_usd"].fillna(0).sum()
                        ),
                        "generation_cost_total_usd": float(
                            summary["generation_cost_total_usd"].fillna(0).sum()
                        ),
                        "total_cost_usd": float(
                            summary["total_cost_usd"].fillna(0).sum()
                        ),
                    },
                )
                log_params(
                    mlflow,
                    {
                        "best_model": str(best_row["model"]),
                        "best_model_tier": str(best_row["tier"]),
                        "timestamp": timestamp,
                    },
                )

        print(f"Détails : {out_path}")
        print(f"Résumé  : {summary_path}")
        return out_path, summary_path
