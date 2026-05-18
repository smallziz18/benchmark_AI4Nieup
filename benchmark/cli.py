"""Interface CLI unique pour le benchmark et le juge."""

from __future__ import annotations

import argparse

from benchmark.catalog import (
    DEFAULT_JUDGE,
    DEFAULT_WEIGHTS,
    JUDGE_MODELS,
    MODEL_CATALOG,
    TIERS,
)
from benchmark.mlflow_utils import MLflowConfig
from benchmark.providers import list_ollama_models, select_models
from benchmark.runner import run_benchmark
from benchmark.scoring import run_judge


def main_benchmark_cli(argv: list[str] | None = None) -> None:
    """Point d'entrée CLI pour exécuter le benchmark de réponses."""
    parser = argparse.ArgumentParser(
        description="L'IA Pour Tous - Benchmark LLMs (Groq, OpenRouter, Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES
  python main.py run --list
  python main.py run --provider openai --models gpt-4o-mini --dataset datasets/dataset_MASTER_10_questions.csv
  python main.py run --provider groq --dataset datasets/dataset_C1_utiliser_IA_grand_public.csv
  python main.py run --provider openrouter --tier frontier --dataset datasets/dataset_MASTER_benchmark_LLM.csv
  python main.py run --provider ollama --dataset datasets/dataset_C1_utiliser_IA_grand_public.csv
        """,
    )
    parser.add_argument("--dataset")
    parser.add_argument("--models")
    parser.add_argument("--tier", choices=TIERS + ["all"])
    parser.add_argument(
        "--provider",
        choices=["openai", "groq", "openrouter", "ollama", "all"],
        default="openrouter",
    )
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--mlflow", action="store_true", help="Active le suivi MLflow")
    parser.add_argument(
        "--mlflow-tracking-uri", help="URI MLflow (ex: http://localhost:5000)"
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="ia_pour_tous_benchmark",
        help="Nom de l'experiment MLflow",
    )
    parser.add_argument("--mlflow-run-name", help="Nom du run MLflow")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--list-tier", choices=TIERS + ["all"])
    parser.add_argument(
        "--list-provider",
        choices=["openai", "groq", "openrouter", "ollama", "all"],
        default="all",
    )
    args = parser.parse_args(argv)

    if args.list or args.list_tier or args.list_provider != "all":
        local_models = list_ollama_models()
        providers = (
            [args.list_provider]
            if args.list_provider != "all"
            else ["openai", "groq", "openrouter", "ollama"]
        )
        print("Catalogue des modeles")
        for prov in providers:
            print(f"\n{prov.upper()}")
            if prov == "ollama":
                for model in local_models:
                    print(f"  {model}")
                continue
            for key, cfg in MODEL_CATALOG.items():
                if cfg["provider"] != prov:
                    continue
                if (
                    args.list_tier not in (None, "all")
                    and cfg["tier"] != args.list_tier
                ):
                    continue
                print(f"  {key:<24} {cfg['display_name']:<25} [{cfg['tier']}]")
        return

    if not args.dataset:
        print("Argument --dataset obligatoire.")
        parser.print_help()
        return

    local_models = list_ollama_models()
    selected = select_models(args.models, args.provider, args.tier, local_models)
    mlflow_config = MLflowConfig(
        enabled=bool(args.mlflow),
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
    )
    run_benchmark(args.dataset, selected, args.output_dir, mlflow_config=mlflow_config)


def main_judge_cli(argv: list[str] | None = None) -> None:
    """Point d'entrée CLI pour exécuter le juge LLM."""
    parser = argparse.ArgumentParser(
        description="L'IA Pour Tous - Juge LLM multi-modèles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES
  python main.py judge --input benchmark_results/responses_XXXX.csv
  python main.py judge --input benchmark_results/responses_XXXX.csv --judges claude-3-sonnet-20240229,groq/llama3-70b-8192
        """,
    )
    parser.add_argument("--input")
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--judges")
    parser.add_argument("--weights", default=",".join(str(x) for x in DEFAULT_WEIGHTS))
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Nombre de workers pour la notation parallèle",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--mlflow", action="store_true", help="Active le suivi MLflow")
    parser.add_argument(
        "--mlflow-tracking-uri", help="URI MLflow (ex: http://localhost:5000)"
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="ia_pour_tous_benchmark",
        help="Nom de l'experiment MLflow",
    )
    parser.add_argument("--mlflow-run-name", help="Nom du run MLflow")
    parser.add_argument("--list-judges", action="store_true")
    args = parser.parse_args(argv)

    if args.list_judges:
        print("Modeles juges disponibles")
        print(f"  {'CLÉ':<24} {'PROVIDER':<11} NOTE")
        for key, info in JUDGE_MODELS.items():
            suffix = " (defaut)" if key == DEFAULT_JUDGE else ""
            print(f"  {key:<24} {info['provider']:<11} {info['note']}{suffix}")
        return

    if not args.input:
        print("Argument --input obligatoire.")
        parser.print_help()
        return

    judge_keys = [
        j.strip() for j in (args.judges or args.judge).split(",") if j.strip()
    ]
    if args.weights:
        parsed = tuple(float(x) for x in args.weights.split(","))
        if len(parsed) != 3:
            raise ValueError("--weights doit contenir exactement 3 valeurs: V,E,A")
        weights = (parsed[0], parsed[1], parsed[2])
    else:
        weights = DEFAULT_WEIGHTS
    mlflow_config = MLflowConfig(
        enabled=bool(args.mlflow),
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
    )
    run_judge(
        args.input,
        judge_keys,
        output_dir=args.output_dir,
        weights=weights,
        workers=args.workers,
        mlflow_config=mlflow_config,
    )
