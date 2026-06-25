# GPT-LIKE

Ce projet pédagogique présente deux implémentations d'un Transformer "decoder-only" de type GPT pour la génération de texte au niveau des caractères.

## Présentation générale

Le projet contient deux scripts principaux :

- `GPT-LIKE.py` : implémentation PyTorch avec autograd et dropout.
- `GPT-LIKE-numpy.py` : implémentation pure NumPy avec backpropagation manuelle.

Ces deux scripts entraînent un modèle de langage sur un petit corpus français et génèrent du texte en utilisant un mécanisme d'attention causale.

## Objectifs pédagogiques

- Comprendre l'architecture d'un Transformer GPT-like.
- Comparer une implémentation PyTorch moderne avec une version NumPy explicite.
- Observer la différence entre entraînement automatique (`autograd`) et rétropropagation manuelle.
- Étudier les composants essentiels : attention, normalisation, résidus, optimisation.

## Détails des deux implémentations

### `GPT-LIKE.py`

- Framework : PyTorch.
- Utilise `torch.nn.Module`, `nn.Linear`, `nn.LayerNorm`, `nn.Dropout`.
- Activation interne : `GELU`.
- Entraînement : `torch.optim.AdamW`.
- Calcul de perte : `F.cross_entropy`.
- Supporte GPU si disponible.
- Génération : attention causale, prise en charge de `top_k` optionnel.

### `GPT-LIKE-numpy.py`

- Framework : NumPy uniquement.
- Implémente manuellement :
  - `Linear`, `LayerNorm`, `Embedding`
  - `MultiHeadAttention` causale
  - `FeedForward` avec ReLU
  - `TransformerBlock` pré-normalisé (pre-LN)
  - `GPT` complet
- Backpropagation implémentée couche par couche.
- Optimiseur : Adam écrit à la main.
- Perte : softmax + cross-entropy avec gradient analytique.
- Exécution CPU uniquement.

## Fonctionnalités communes

Les deux scripts partagent les concepts suivants :

- Embedding de tokens et positionnel.
- Attention multi-têtes causale.
- Architecture `pre-LN` comparable à GPT-2.
- Résidus sur chaque sous-bloc.
- Modèle de langage autoregressif.
- Boucle d'entraînement sur un corpus fixe.
- Génération de texte après entraînement.

## Prérequis

- Python 3.8 ou supérieur.
- `numpy` pour l'implémentation NumPy.
- `torch` pour l'implémentation PyTorch.

## Installation

1. Ouvre un terminal dans le dossier du projet.
2. (Optionnel) Crée un environnement virtuel :

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Installe les dépendances :

```powershell
pip install numpy torch
```

## Exécution

### Exécuter le modèle PyTorch

```powershell
python GPT-LIKE.py
```

### Exécuter le modèle NumPy

```powershell
python GPT-LIKE-numpy.py
```

## Résultat attendu

- Le script affiche les informations de taille du vocabulaire et le nombre total de paramètres.
- Il affiche la perte pendant l'entraînement.
- Il génère ensuite un texte à partir d'un prompt de départ.

## Personnalisation

Tu peux adapter les scripts en modifiant les paramètres suivants :

- `TEXT` : corpus d'entraînement.
- `d_model`, `n_heads`, `n_layers`, `d_ff` : dimensions du modèle.
- `block_size`, `batch_size`, `n_steps`, `lr` : hyperparamètres d'entraînement.
- `temperature`, `start_str` / `max_new_tokens` : génération de texte.

## Comparaison rapide

- `GPT-LIKE.py` : plus proche d'une implémentation industrielle, exploitation de PyTorch, plus court et optimisé.
- `GPT-LIKE-numpy.py` : version didactique, explicite, sans dépendance à un framework de deep learning.

## Remarques

- `GPT-LIKE-numpy.py` est utile pour comprendre chaque étape de calcul et de mise à jour des gradients.
- `GPT-LIKE.py` est utile pour comparer le code pédagogique à une implémentation réelle avec une API standard.
- Aucun des deux scripts n'est destiné à un usage de production : ce sont des outils d'apprentissage.

## Structure du projet

- `GPT-LIKE.py` : implémentation PyTorch.
- `GPT-LIKE-numpy.py` : implémentation NumPy.
- `README.md` : documentation du projet.