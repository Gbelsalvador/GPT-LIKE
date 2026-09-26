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

## Organisation

- `script.py` : point d'entree CLI.
- `config.py` : configurations legere et 124M.
- `dataset.py` : dataset causal et DataLoaders train/validation.
- `GPT.py` : modele GPT-like PyTorch.
- `loss.py` : cross-entropy et evaluation.
- `train.py` : boucle d'entrainement.
- `generate.py` : generation autoregressive.
- `tokenizer.py` : conversion texte/tokens avec tiktoken.
- `architecture/` : implementations pedagogiques conservees telles quelles.

La configuration legere est utilisee par defaut afin de fonctionner sur CPU.
