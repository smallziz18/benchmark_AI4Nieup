# L'IA Pour Tous - Benchmark LLM

Benchmark de modeles LLM pour tutorat socratique, sur les datasets C1 a C5.

## Installation rapide

```bash
pip install -r requirements.txt
python check_setup.py
```

Variables utiles:
- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `OLLAMA_HOST` (optionnel, defaut `http://localhost:11434`)

## Suivi MLflow

Le benchmark peut maintenant etre trace dans MLflow pour garder un historique propre des runs.

Exemples:

```bash
python benchmark_runner.py \
  --dataset datasets/dataset_C1_utiliser_IA_grand_public.csv \
  --provider groq \
  --mlflow
```

```bash
python judge.py \
  --input benchmark_results/responses_XXXX.csv \
  --judge gpt-5.4-mini \
  --mlflow
```

Options utiles:
- `--mlflow-tracking-uri` : ex. `http://localhost:5000`
- `--mlflow-experiment` : nom de l'experiment (defaut `ia_pour_tous_benchmark`)
- `--mlflow-run-name` : nom lisible du run dans MLflow

Artefacts logs:
- `responses_*.csv`
- `scored_*.csv`
- `summary_*.csv`
- fichiers CSV par modele / juges quand disponibles

## Commandes principales

Lister les modeles:

```bash
python benchmark_runner.py --list
```

Lancer un benchmark (exemple C1 avec Groq):

```bash
python benchmark_runner.py \
  --dataset datasets/dataset_C1_utiliser_IA_grand_public.csv \
  --provider groq
```

Re-noter uniquement avec ensemble scoring (exemple OpenAI direct):

```bash
python judge.py \
  --input benchmark_results/responses_XXXX.csv \
  --judges gpt-5.4-mini,gpt-5.4-openai,claude-sonnet-4.6,gemini-3.1-pro,deepseek-r1
```

Noter les reponses:

```bash
python judge.py --input benchmark_results/responses_XXXX.csv
```

## Exemples prets a lancer

Voir `examples/`:
- `examples/list_models.sh`
- `examples/run_benchmark_c1_groq.sh`
- `examples/run_judge_latest.sh`

## Sorties generees

- `benchmark_results/responses_YYYYMMDD_HHMMSS.csv`
- `benchmark_results/scored_YYYYMMDD_HHMMSS.csv`
- `benchmark_results/summary_YYYYMMDD_HHMMSS.csv`

Le `dashboard.html` lit directement les CSV de score.

## Architecture

- `benchmark/catalog.py`: catalogue modeles et poids
- `benchmark/prompts.py`: prompts benchmark/juge
- `benchmark/providers.py`: clients Groq/OpenRouter/Ollama
- `benchmark/runner.py`: generation des reponses
- `benchmark/scoring.py`: evaluation par juge LLM
- `benchmark/aggregation.py`: classement et robustesse
- `benchmark/utils.py`: utilitaires partages
- `benchmark/cli.py`: interface CLI

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Contribution

- Guide de contribution: `CONTRIBUTING.md`
- Architecture: `docs/ARCHITECTURE.md`
- Methodologie benchmark (vue globale): `docs/METHODOLOGIE_BENCHMARK.md`
- Conventions de code: `docs/CONVENTIONS.md`
- Strategie de tests: `docs/TESTING.md`
- Dictionnaire des metriques `summary_*.csv`: `docs/README_METRIQUES_SUMMARY.md`

Pour des commandes prêtes a lancer, voir `examples/README.md`.
