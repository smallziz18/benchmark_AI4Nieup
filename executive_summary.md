# Résumé Exécutif du Benchmark LLM IA Pour Tous

## 1. Ce qu’on a testé, pourquoi, et les 3 grandes conclusions

Ce benchmark évalue 15 modèles LLM open source (Ollama local) sur plus de 300 questions couvrant 5 compétences clés de l’IA pour le grand public. L’objectif est de comparer la qualité pédagogique des réponses générées, en utilisant une notation automatisée par 8 juges LLM. 

**3 grandes conclusions :**
1. Les modèles open source récents atteignent désormais un niveau de validité et d’étayage proche des modèles propriétaires sur des tâches de vulgarisation.
2. Les différences majeures se situent sur l’adaptation au niveau de l’utilisateur et la capacité à éviter les hallucinations.
3. L’automatisation de la notation par LLM permet un benchmark massif, mais nécessite une vigilance sur la robustesse des juges et la diversité des prompts.

## 2. Contexte & objectifs

**Problème résolu :**
- Fournir une évaluation objective et reproductible de la capacité des LLM à répondre à des questions d’IA grand public, en français.
- Identifier les modèles les plus adaptés à l’enseignement, la médiation scientifique et l’auto-formation.

**Pourquoi ces modèles ?**
- Sélection de 15 modèles open source populaires et récents, disponibles localement via Ollama (ex : Qwen3, Llama3, Mistral, Gemma, Deepseek, etc.).
- Représentation de différentes familles (Meta, Google, Alibaba, Mistral, etc.) et tailles (7B à 70B).

**Pourquoi ces compétences ?**
- 5 axes de compétences : utiliser, analyser, construire, innover avec l’IA, et mathématiques pour l’IA.
- Couvre l’ensemble du référentiel de l’IA pour tous (usages, compréhension, création, esprit critique).

## 3. Méthodologie

**Dimensions évaluées :**
- **Validité (V)** : exactitude, absence d’erreur ou d’hallucination.
- **Clarté/Étayage (C/E)** : capacité à expliquer, justifier, argumenter.
- **Adaptation (A)** : pertinence de la réponse au niveau et au contexte de l’utilisateur.

**LLM-as-a-judge :**
- Chaque réponse est notée par 8 juges LLM indépendants, selon un prompt standardisé.
- Les scores sont agrégés pour chaque critère (V/E/A).
- Limites : risque de biais, sur-apprentissage du prompt, honnêteté académique (le juge peut être trompé par une réponse bien formulée mais fausse).

**Conditions d’évaluation :**
- 5314 réponses générées (15 modèles × 300+ questions).
- 8 juges LLM locaux (Ollama) pour chaque réponse.
- Parallélisation sur machine locale (Ryzen 7 6800H, 32 Go RAM, RTX 3070 Ti).
- Notation automatisée, reproductible, logs et résultats archivés.

---

Pour plus de détails, voir le protocole complet et les résultats agrégés dans le dossier du projet.
