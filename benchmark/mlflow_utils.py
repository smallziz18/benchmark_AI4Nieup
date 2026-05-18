"""Utilitaires MLflow pour tracer les runs du benchmark."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Mapping, Optional

import pandas as pd


@dataclass(frozen=True)
class MLflowConfig:
    """Configuration simple pour activer ou non le suivi MLflow."""

    enabled: bool = False
    tracking_uri: Optional[str] = None
    experiment_name: str = "ia_pour_tous_benchmark"
    run_name: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)


def load_mlflow():
    """Charge `mlflow` si le package est disponible, sinon retourne `None`."""
    try:
        import mlflow  # type: ignore
    except Exception:
        return None
    return mlflow


def _normalize_value(value: Any, max_len: int = 500) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text[:max_len]


def _clean_mapping(values: Mapping[str, Any] | None) -> dict[str, str]:
    if not values:
        return {}
    return {str(k): _normalize_value(v) for k, v in values.items() if v is not None}


def configure_mlflow(config: Optional[MLflowConfig]):
    """Prépare MLflow et retourne le module ou `None` si non activé/disponible."""
    if not config or not config.enabled:
        return None
    mlflow = load_mlflow()
    if mlflow is None:
        return None
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    if config.experiment_name:
        mlflow.set_experiment(config.experiment_name)
    return mlflow


@contextmanager
def start_mlflow_run(
    config: Optional[MLflowConfig],
    *,
    nested: bool = False,
    run_name: Optional[str] = None,
) -> Iterator[Any]:
    """Ouvre un run MLflow si l'option est active, sinon un contexte vide."""
    mlflow = configure_mlflow(config)
    if mlflow is None:
        yield None
        return

    with mlflow.start_run(
        run_name=(
            run_name if run_name is not None else (config.run_name if config else None)
        ),
        nested=nested,
    ):
        if config and config.tags:
            mlflow.set_tags(_clean_mapping(config.tags))
        yield mlflow


def log_params(mlflow: Any, params: Mapping[str, Any] | None) -> None:
    """Log des paramètres MLflow en ignorant les valeurs vides."""
    if mlflow is None:
        return
    for key, value in _clean_mapping(params).items():
        mlflow.log_param(key, value)


def log_metrics(mlflow: Any, metrics: Mapping[str, Any] | None) -> None:
    """Log des métriques MLflow en ignorant les valeurs non numériques."""
    if mlflow is None:
        return
    for key, value in (metrics or {}).items():
        if value is None:
            continue
        try:
            mlflow.log_metric(str(key), float(value))
        except Exception:
            continue


def log_artifact_path(
    mlflow: Any, path: str | Path, artifact_path: Optional[str] = None
) -> None:
    """Log un artefact existant si MLflow est actif."""
    if mlflow is None:
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_dataframe(
    mlflow: Any, df: pd.DataFrame, filename: str, artifact_path: Optional[str] = None
) -> None:
    """Sérialise un DataFrame dans un CSV temporaire puis le log comme artefact."""
    if mlflow is None:
        return
    with NamedTemporaryFile(
        mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
        df.to_csv(tmp_path, index=False, encoding="utf-8")
    try:
        log_artifact_path(mlflow, tmp_path, artifact_path=artifact_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
