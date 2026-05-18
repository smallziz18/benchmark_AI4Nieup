import argparse

from benchmark.cli import main_benchmark_cli, main_judge_cli


def main():
    """
    Point d'entrée principal qui délègue aux sous-commandes `run` ou `judge`.
    """
    parser = argparse.ArgumentParser(
        description="""L'IA Pour Tous - Outil de Benchmark pour Modèles de Langage.

Ce CLI unifié permet de lancer des benchmarks de génération de réponses (`run`) 
et d'évaluation de ces réponses par des juges LLM (`judge`).
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Sous-commande à exécuter")
    subparsers.required = True

    # Commande 'run'
    parser_run = subparsers.add_parser(
        "run",
        help="Lancer la génération de réponses par les modèles.",
        description="Exécute le benchmark de génération de réponses sur un jeu de données.",
        epilog="""
EXEMPLES D'UTILISATION :
  # Lister tous les modèles disponibles
  python main.py run --list

  # Lancer un benchmark sur le dataset C1 avec les modèles de Groq
  python main.py run --provider groq --dataset datasets/dataset_C1_utiliser_IA_grand_public.csv

  # Lancer un benchmark avec des modèles spécifiques
  python main.py run --models gpt-4o-mini,claude-3-haiku-20240307 --dataset datasets/dataset_MASTER_10_questions.csv
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_run.set_defaults(func=main_benchmark_cli)

    # Commande 'judge'
    parser_judge = subparsers.add_parser(
        "judge",
        help="Lancer l'évaluation des réponses par un ou plusieurs juges.",
        description="Évalue les réponses générées (fichier CSV) à l'aide de modèles juges.",
        epilog="""
EXEMPLES D'UTILISATION :
  # Lister les modèles juges disponibles
  python main.py judge --list-judges

  # Évaluer un fichier de réponses avec le juge par défaut
  python main.py judge --input benchmark_results/responses_XYZ.csv

  # Évaluer avec plusieurs juges spécifiques
  python main.py judge --input benchmark_results/responses_XYZ.csv --judges gpt-4o-mini,claude-3-sonnet-20240229
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_judge.set_defaults(func=main_judge_cli)

    # Analyser les arguments de la ligne de commande
    # On ne parse que le premier niveau pour identifier la sous-commande
    # Le parsing complet sera fait par les fonctions déléguées
    args, remaining_argv = parser.parse_known_args()

    # Appeler la fonction associée à la sous-commande choisie
    args.func(remaining_argv)


if __name__ == "__main__":
    main()
