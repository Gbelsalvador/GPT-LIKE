# GPT-like en PyTorch

Projet pedagogique qui organise une implementation de modele de langage autoregressif de type GPT-2 dans des modules Python reutilisables.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch tiktoken
```

## Lancer un entrainement

```powershell
python script.py
python script.py --text corpus.txt --epochs 10 --prompt "je suis" --save model.pt
```

Le corpus par defaut est `data/train.txt`. Utilise `python script.py --text autre_corpus.txt` pour en charger un autre, ou `python script.py --help` pour les options du device, du batch, du contexte et du taux d'apprentissage.

## Generer avec un modele enregistre

```powershell
python script.py --generate --prompt "je suis" --max-new-tokens 100
python script.py --generate --model chemin/vers/modele.pt --prompt "je suis" --max-new-tokens 100
```

Sans `--model`, le script charge `model.pt` a la racine du projet. Entraine et enregistre d'abord le modele avec `--save model.pt` si ce fichier n'existe pas encore.

## Fine-tuning supervisé

Après le pré-entraînement, `finetuner.py` adapte un checkpoint existant à des exemples d'instructions et de réponses. Le dataset par défaut est `data/instructions.jsonl` ; chaque ligne est un objet JSON avec uniquement deux champs : `instruction` et `output`.

```powershell
python finetuner.py --model model.pt --data data/instructions.jsonl --epochs 3 --save finetuned_model.pt
```

Le checkpoint obtenu conserve le format attendu par `script.py --generate`. La perte est calculée sur `output`, pas sur le texte de l'instruction. Remplace ou complète le dataset d'exemple par des données représentatives de la tâche visée.

### Fine-tuning de classification

Le mode classification apprend à associer un texte à une étiquette. Le dataset fourni (`data/classification.jsonl`) contient des exemples français de sentiment avec les champs `text` et `label`. Il sert de démonstration; pour un usage réel, remplace-le par des exemples représentatifs et suffisamment nombreux.

```powershell
python finetuner.py --task classification --model model.pt --epochs 3
python classify.py --model classification_model.pt --text "Le service était excellent."
```

Le mode classification crée une tête de classification au-dessus du GPT, entraîne aussi le backbone et enregistre séparément `classification_model.pt`. Les sorties donnent l'étiquette prédite et sa confiance. Le checkpoint de classification n'est pas un checkpoint de génération.

## Organisation

- `script.py` : point d'entree CLI.
- `config.py` : configurations legere et 124M.
- `dataset.py` : dataset causal et DataLoaders train/validation.
- `GPT.py` : modele GPT-like PyTorch.
- `loss.py` : cross-entropy et evaluation.
- `train.py` : boucle d'entrainement.
- `finetuner.py` : fine-tuning supervisé pour les instructions ou la classification.
- `classify.py` : prédiction avec un checkpoint fine-tuné pour la classification.
- `generate.py` : generation autoregressive.
- `tokenizer.py` : conversion texte/tokens avec tiktoken.
- `architecture/` : implementations pedagogiques conservees telles quelles.

La configuration legere est utilisee par defaut afin de fonctionner sur CPU.
