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

Utilise `python script.py --help` pour les options du device, du batch, du contexte et du taux d'apprentissage.

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
