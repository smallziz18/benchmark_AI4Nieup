# Contributing

Merci de contribuer au benchmark.

## Workflow rapide

1. Creer une branche depuis `main`.
2. Faire des changements petits et atomiques.
3. Lancer les tests localement.
4. Ouvrir une Pull Request avec une description claire.

## Standards du projet

- Python 3.9+
- Code lisible, simple, DRY
- Pas de duplication de logique entre modules
- Docstrings sur les fonctions publiques
- Pas de changement de schema CSV sans justification

## Commandes a executer avant PR

```bash
python -m unittest discover -s tests -p "test_*.py"
python benchmark_runner.py --list
python judge.py --list-judges
```

## Structure a respecter

- `benchmark/catalog.py`: modeles et poids
- `benchmark/prompts.py`: prompts
- `benchmark/providers.py`: clients API et selection de modeles
- `benchmark/runner.py`: generation de reponses
- `benchmark/scoring.py`: evaluation et scoring
- `benchmark/aggregation.py`: ranking et robustesse
- `benchmark/utils.py`: fonctions utilitaires pures
- `benchmark/cli.py`: couche CLI

## Convention de commit

Format conseille:

- `feat: ...` pour une fonctionnalite
- `fix: ...` pour une correction
- `docs: ...` pour documentation
- `refactor: ...` pour refactor sans changement fonctionnel
- `test: ...` pour ajout/modif de tests

## Checklist PR

- [ ] Code documente
- [ ] Tests ajoutes/maj
- [ ] Tests locaux passants
- [ ] README ou docs mises a jour si necessaire
- [ ] Changement compatible avec `dashboard.html`

