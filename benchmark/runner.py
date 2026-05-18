"""Exécution du benchmark réponse par réponse."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from benchmark.mlflow_utils import (
    MLflowConfig,
    log_artifact_path,
    log_dataframe,
    log_metrics,
    log_params,
    start_mlflow_run,
)
from benchmark.prompts import SYSTEM_PROMPT
from benchmark.providers import (
    detect_provider,
    estimate_cost_usd,
    get_model_pricing_usd_per_1m,
    get_openai_client,
    list_ollama_models,
)


def _chat_completion(
    client: OpenAI,
    provider: str,
    model_id: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool = False,
    top_k: int | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Envoie une completion chat avec fallback non-stream en cas d'erreur stream."""
    extra_headers = None
    if provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": "https://ia-pour-tous.sn",
            "X-Title": "IA Pour Tous Benchmark",
        }

    request_kwargs: Dict[str, Any] = {
        "model": model_id,
        "messages": cast(Any, messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_headers": extra_headers,
    }
    # top_k n'est pas supporté nativement par l'API OpenAI directe.
    if top_k is not None and provider in {"openrouter", "ollama"}:
        request_kwargs["extra_body"] = {"top_k": int(top_k)}

    if not stream:
        resp = client.chat.completions.create(**request_kwargs)  # type: ignore[arg-type]
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return content, {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "ttft_s": None,
            "latency_s": None,
        }

    start = time.time()
    first_token_at = None
    chunks: List[str] = []
    usage = None
    try:
        stream_kwargs = {
            **request_kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream_resp = client.chat.completions.create(**stream_kwargs)  # type: ignore[arg-type]
        for event in stream_resp:
            choice = event.choices[0] if getattr(event, "choices", None) else None
            if choice and getattr(choice, "delta", None):
                content = getattr(choice.delta, "content", None)
                if isinstance(content, str) and content:
                    chunks.append(content)
                    if first_token_at is None:
                        first_token_at = time.time() - start
            if getattr(event, "usage", None):
                usage = event.usage
    except Exception:
        content, meta = _chat_completion(
            client,
            provider,
            model_id,
            messages,
            temperature,
            max_tokens,
            stream=False,
            top_k=top_k,
        )
        meta["latency_s"] = round(time.time() - start, 2)
        return content, meta

    return "".join(chunks), {
        "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "ttft_s": round(first_token_at, 3) if first_token_at is not None else None,
        "latency_s": round(time.time() - start, 2),
    }


def benchmark_question(
    client: OpenAI,
    provider: str,
    model_id: str,
    question: str,
    niveau: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_k: int | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Génère la réponse d'un modèle pour une question du dataset."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[Niveau : {niveau}]\n\n{question}"},
    ]
    return _chat_completion(
        client,
        provider,
        model_id,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        top_k=top_k,
    )


def _response_row_from_dataset(row: pd.Series) -> Dict[str, Any]:
    """Mappe une ligne dataset vers le schéma standard des réponses."""
    return {
        "id": int(row["ID"]),
        "code_competence": row.get("Code_Competence", row.get("code_competence", "")),
        "competence": row.get("Competence", row.get("competence", "")),
        "niveau": row.get("Niveau", row.get("niveau", "")),
        "difficulte": row.get("Difficulte", row.get("difficulte", "")),
        "sujet": row.get("Sujet", row.get("sujet", "")),
        "question": row.get("Question_Etudiant", row.get("question", "")),
        "reponse_ideale": row.get(
            "Reponse_Ideale_Tuteur", row.get("reponse_ideale", "")
        ),
        "critere": row.get("Critere_Evaluation", row.get("critere", "")),
    }


def run_benchmark(
    dataset_path: str,
    models: Sequence[str],
    output_dir: str,
    mlflow_config: MLflowConfig | None = None,
) -> str:
    """Exécute le benchmark et écrit `responses_*.csv`.

    Retourne le chemin du fichier généré.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)  # type: ignore[call-overload]
    required_cols = [
        "ID",
        "Competence",
        "Niveau",
        "Sujet",
        "Question_Etudiant",
        "Reponse_Ideale_Tuteur",
        "Critere_Evaluation",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le dataset: {missing}. Colonnes trouvées: {list(df.columns)}"
        )

    local_models = list_ollama_models()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results: List[Dict[str, Any]] = []
    mlflow_parent_name = (
        mlflow_config.run_name
        if mlflow_config and mlflow_config.run_name
        else f"benchmark:{Path(dataset_path).stem}:{timestamp}"
    )

    print("\nBenchmark runner")
    print(f"Dataset   : {os.path.basename(dataset_path)}")
    print(f"Questions : {len(df)}")
    print(f"Modeles   : {len(models)}")

    with start_mlflow_run(
        mlflow_config, nested=False, run_name=mlflow_parent_name
    ) as mlflow:
        if mlflow is not None:
            log_params(
                mlflow,
                {
                    "stage": "benchmark",
                    "dataset_path": dataset_path,
                    "dataset_name": Path(dataset_path).name,
                    "output_dir": output_dir,
                    "models": ",".join(models),
                    "nb_models": len(models),
                    "nb_questions": len(df),
                },
            )
            log_metrics(mlflow, {"nb_models": len(models), "nb_questions": len(df)})

        for model_key in models:
            resolved_key, provider, model_id, cfg = detect_provider(
                model_key, local_models
            )
            display_name = cfg.get("display_name", model_key)
            pricing = get_model_pricing_usd_per_1m(provider, model_id)
            top_k = int(cfg["top_k"]) if cfg.get("top_k") is not None else None
            print(f"\nModel: {display_name} [{provider}]")
            client = get_openai_client(provider)

            model_rows: List[Dict[str, Any]] = []
            for _, row in tqdm(
                df.iterrows(),
                total=len(df),
                ncols=78,
                desc=f"  {display_name[:22]:<22}",
            ):
                start = time.time()
                try:
                    response, meta = benchmark_question(
                        client,
                        provider,
                        model_id,
                        row["Question_Etudiant"],
                        row["Niveau"],
                        top_k=top_k,
                    )
                    latency_s = (
                        meta.get("latency_s")
                        if meta.get("latency_s") is not None
                        else round(time.time() - start, 2)
                    )
                    input_tokens = meta.get("input_tokens")
                    output_tokens = meta.get("output_tokens")
                    ttft_s = meta.get("ttft_s")
                    tpot_s = (
                        round(latency_s / output_tokens, 4)
                        if isinstance(output_tokens, int)
                        and output_tokens > 0
                        and latency_s
                        else None
                    )
                    estimated_cost_usd = estimate_cost_usd(
                        provider, model_id, input_tokens, output_tokens
                    )
                    error = None
                except Exception as exc:
                    response = ""
                    latency_s = round(time.time() - start, 2)
                    input_tokens = None
                    output_tokens = None
                    ttft_s = None
                    tpot_s = None
                    estimated_cost_usd = None
                    error = str(exc)

                row_out = {
                    **_response_row_from_dataset(row),
                    "model_key": resolved_key,
                    "model_name": display_name,
                    "provider": provider,
                    "tier": cfg.get(
                        "tier", "local" if provider == "ollama" else "unknown"
                    ),
                    "response": response,
                    "latency_s": latency_s,
                    "ttft_s": ttft_s,
                    "tpot_s": tpot_s,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                    "price_input_usd_per_1m": pricing["input"] if pricing else None,
                    "price_output_usd_per_1m": pricing["output"] if pricing else None,
                    "error": error,
                    "score": None,
                    "score_V": None,
                    "score_E": None,
                    "score_A": None,
                    "score_details": None,
                }
                results.append(row_out)
                model_rows.append(row_out)

            if mlflow is not None:
                with mlflow.start_run(run_name=f"{resolved_key}", nested=True):
                    model_df = pd.DataFrame(model_rows)
                    valid_df = model_df[model_df["error"].isna()].copy()
                    lat_series = (
                        valid_df["latency_s"].dropna()
                        if not valid_df.empty
                        else pd.Series(dtype=float)
                    )
                    ttft_series = (
                        valid_df["ttft_s"].dropna()
                        if not valid_df.empty
                        else pd.Series(dtype=float)
                    )
                    cost_series = (
                        valid_df["estimated_cost_usd"].dropna()
                        if not valid_df.empty
                        else pd.Series(dtype=float)
                    )
                    log_params(
                        mlflow,
                        {
                            "stage": "benchmark-model",
                            "model_key": resolved_key,
                            "model_name": display_name,
                            "provider": provider,
                            "model_id": model_id,
                            "tier": cfg.get("tier", "unknown"),
                            "top_k": top_k,
                            "display_name": display_name,
                        },
                    )
                    log_metrics(
                        mlflow,
                        {
                            "nb_questions": len(model_df),
                            "nb_errors": int(model_df["error"].notna().sum()),
                            "latence_moy_s": (
                                float(lat_series.mean())
                                if not lat_series.empty
                                else None
                            ),
                            "latence_med_s": (
                                float(lat_series.median())
                                if not lat_series.empty
                                else None
                            ),
                            "ttft_moy_s": (
                                float(ttft_series.mean())
                                if not ttft_series.empty
                                else None
                            ),
                            "estimated_cost_total_usd": (
                                float(cost_series.sum())
                                if not cost_series.empty
                                else None
                            ),
                            "estimated_cost_per_question_usd": (
                                float(cost_series.mean())
                                if not cost_series.empty
                                else None
                            ),
                        },
                    )
                    log_dataframe(
                        mlflow,
                        model_df,
                        f"responses_{resolved_key}.csv",
                        artifact_path="per_model",
                    )

    out_df = pd.DataFrame(results)
    out_path = str(Path(output_dir) / f"responses_{timestamp}.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    if mlflow is not None:
        log_artifact_path(mlflow, out_path, artifact_path="benchmark_outputs")
        valid_all = out_df[out_df["error"].isna()].copy()
        log_metrics(
            mlflow,
            {
                "nb_reponses": float(len(out_df)),
                "nb_reponses_valides": float(len(valid_all)),
                "nb_erreurs": float(out_df["error"].notna().sum()),
                "latence_moyenne_tous_modeles": (
                    float(valid_all["latency_s"].mean())
                    if not valid_all.empty
                    else None
                ),
                "cout_estime_total_usd": (
                    float(valid_all["estimated_cost_usd"].dropna().sum())
                    if not valid_all.empty
                    else None
                ),
            },
        )
    print(f"Responses saved: {out_path}")
    return out_path
