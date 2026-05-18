"""Providers OpenAI, Groq, OpenRouter et Ollama."""

from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from openai import OpenAI

from benchmark.catalog import MODEL_CATALOG

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GROQ_BASE = "https://api.groq.com/openai/v1"
OPENAI_BASE = "https://api.openai.com/v1"
OLLAMA_DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def normalize_ollama_base_host(host: Optional[str] = None) -> str:
    """Retourne l'hôte Ollama normalisé pour un accès client local.

    Accepte les valeurs brutes comme `0.0.0.0:11434` et les ramène vers
    `127.0.0.1`, qui est utilisable depuis le client sur Windows.
    """
    raw = (host or os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_HOST)).strip()
    if not raw:
        raw = "http://127.0.0.1:11434"
    if "//" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    hostname = parts.hostname or ""
    if hostname == "0.0.0.0":
        netloc = "127.0.0.1"
        if parts.port:
            netloc += f":{parts.port}"
        parts = parts._replace(netloc=netloc)
    return urlunsplit(parts).rstrip("/")


@lru_cache(maxsize=1)
def _openrouter_pricing_map() -> Dict[str, Dict[str, float]]:
    """Charge le pricing OpenRouter et le convertit en USD / 1M tokens.

    Retourne un mapping `model_id -> {input, output}`.
    """
    pricing: Dict[str, Dict[str, float]] = {}
    try:
        r = requests.get(f"{OPENROUTER_BASE}/models", timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        for item in data:
            model_id = item.get("id")
            if not model_id:
                continue
            p = item.get("pricing", {})
            in_token = float(p.get("prompt") or 0.0)
            out_token = float(p.get("completion") or 0.0)
            # L'API retourne des prix par token -> conversion / 1M tokens.
            pricing[model_id] = {
                "input": in_token * 1_000_000,
                "output": out_token * 1_000_000,
            }
    except Exception:
        return {}
    return pricing


def get_model_pricing_usd_per_1m(
    provider: str, model_id: str
) -> Optional[Dict[str, float]]:
    """Retourne le pricing d'un modèle en USD / 1M tokens.

    Pour OpenRouter, le pricing est récupéré dynamiquement via l'API `/models`.
    Pour les autres providers, retourne `None` si non disponible.
    """
    provider = provider.lower()
    if provider == "openrouter":
        return _openrouter_pricing_map().get(model_id)
    # Ollama local: pas de coût API.
    if provider == "ollama":
        return {"input": 0.0, "output": 0.0}
    # Groq: non géré ici pour éviter des hypothèses de pricing.
    return None


def estimate_cost_usd(
    provider: str,
    model_id: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    """Calcule un coût estimé en USD à partir des tokens et du pricing provider."""
    pricing = get_model_pricing_usd_per_1m(provider, model_id)
    if not pricing:
        return None
    in_tok = int(input_tokens or 0)
    out_tok = int(output_tokens or 0)
    cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000
    return round(cost, 6)


def normalize_ollama_host(host: Optional[str] = None) -> str:
    """Retourne l'URL base Ollama au format compatible OpenAI (`.../v1`)."""
    base = normalize_ollama_base_host(host)
    return base if base.endswith("/v1") else base + "/v1"


def normalize_ollama_tags_url(host: Optional[str] = None) -> str:
    """Retourne l'URL des tags Ollama (`.../api/tags`)."""
    base = normalize_ollama_base_host(host)
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/api/tags"


def get_openai_client(provider: str) -> OpenAI:
    """Construit un client OpenAI compatible avec le provider choisi.

    Providers supportés: `openai`, `openrouter`, `groq`, `ollama`.
    """
    provider = provider.lower()
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY non définie")
        return OpenAI(base_url=OPENAI_BASE, api_key=key)
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY non définie")
        return OpenAI(base_url=OPENROUTER_BASE, api_key=key)
    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("GROQ_API_KEY non définie")
        return OpenAI(base_url=GROQ_BASE, api_key=key)
    if provider == "ollama":
        return OpenAI(
            base_url=normalize_ollama_host(),
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        )
    raise ValueError(f"Provider inconnu: {provider}")


def list_ollama_models(host: Optional[str] = None) -> List[str]:
    """Liste les modèles locaux disponibles via Ollama.

    Retourne une liste vide si Ollama est indisponible.
    """
    try:
        r = requests.get(normalize_ollama_tags_url(host), timeout=3)
        r.raise_for_status()
        models = r.json().get("models", [])
        return [m.get("name", "") for m in models if m.get("name")]
    except Exception:
        return []


def detect_provider(
    model_key: str, local_models: Sequence[str]
) -> Tuple[str, str, str, Dict[str, Any]]:
    """Résout un `model_key` en provider, model_id et metadata.

    Lève `KeyError` si le modèle est inconnu.
    """
    # Cas catalogue connu
    if model_key in MODEL_CATALOG:
        cfg = MODEL_CATALOG[model_key]
        return model_key, cfg["provider"], cfg["model_id"], cfg

    # Cas modèle local Ollama déjà découvert
    if model_key in local_models:
        return (
            model_key,
            "ollama",
            model_key,
            {
                "provider": "ollama",
                "model_id": model_key,
                "display_name": model_key,
                "family": "Ollama",
                "tier": "local",
                "context_k": None,
                "note": "Modèle local Ollama détecté dynamiquement",
            },
        )

    # Cas Ollama brut non listé explicitement, ex: "qwen3:8b".
    # Si aucun provider explicite ne correspond, on le traite comme un modèle local.
    if ":" in model_key:
        prefix, _, remainder = model_key.partition(":")
        if (
            prefix.lower() not in {"openai", "openrouter", "groq", "ollama"}
            and remainder
        ):
            return (
                model_key,
                "ollama",
                model_key,
                {
                    "provider": "ollama",
                    "model_id": model_key,
                    "display_name": model_key,
                    "family": "Ollama",
                    "tier": "local",
                    "context_k": None,
                    "note": "Modèle local Ollama déduit depuis la CLI",
                },
            )

    # Support d'un model_key préfixé par le provider, ex: "ollama:llama3.1:8b"
    if ":" in model_key:
        prefix, _, remainder = model_key.partition(":")
        prefix_l = prefix.lower()
        if prefix_l in {"openai", "openrouter", "groq", "ollama"}:
            provider = prefix_l
            model_id = remainder
            cfg = {
                "provider": provider,
                "model_id": model_id,
                "display_name": model_id,
                "family": "Custom",
                "tier": "local" if provider == "ollama" else "unknown",
                "context_k": None,
                "note": "Modèle spécifié directement via la CLI",
            }
            return model_key, provider, model_id, cfg

    raise KeyError(f"Modèle inconnu: {model_key}")


def select_models(
    models_arg: Optional[str],
    provider: str,
    tier: Optional[str],
    local_models: Sequence[str],
) -> List[str]:
    """Sélectionne les modèles à lancer selon les options CLI.

    - `models_arg` prioritaire si fourni
    - sinon sélection par `provider` + `tier`
    """
    if models_arg:
        raw_selected = [m.strip() for m in models_arg.split(",") if m.strip()]
        selected: List[str] = []
        unknown: List[str] = []
        for m in raw_selected:
            if m in MODEL_CATALOG or m in local_models:
                selected.append(m)
                continue
            # Si un provider explicite est fourni (non 'all'), autorise les IDs bruts
            if provider != "all":
                # Préfixe le modèle par le provider pour que detect_provider puisse l'interpréter
                selected.append(f"{provider}:{m}")
                continue
            unknown.append(m)

        if unknown:
            raise KeyError(f"Modèles inconnus: {unknown}")
        return selected

    providers = (
        [provider] if provider != "all" else ["openai", "groq", "openrouter", "ollama"]
    )
    selected: List[str] = []
    for key, cfg in MODEL_CATALOG.items():
        if cfg["provider"] in providers and (
            tier is None or tier == "all" or cfg["tier"] == tier
        ):
            selected.append(key)
    if "ollama" in providers and (tier is None or tier == "all"):
        selected.extend([m for m in local_models if m not in selected])
    return selected
