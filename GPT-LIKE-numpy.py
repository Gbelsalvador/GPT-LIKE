"""
================================================================================
TRANSFORMER FROM SCRATCH - 100% NumPy (sans PyTorch / TensorFlow / JAX)
================================================================================

Ce script implémente un Transformer de type "decoder-only" (architecture
GPT-like, autoregressive, causal self-attention), avec :

  - Embedding de tokens + embedding positionnel
  - Multi-Head Self-Attention (causale)
  - Feed-Forward (MLP) avec ReLU
  - LayerNorm
  - Connexions résiduelles (architecture "pre-LN", comme GPT-2)
  - Tête de sortie (projection vers le vocabulaire) + softmax cross-entropy
  - Backpropagation manuelle (calcul des gradients à la main, couche par couche)
  - Optimiseur Adam (implémenté à la main)
  - Boucle d'entraînement sur un mini corpus de texte (char-level language model)
  - Génération de texte (sampling) après entraînement

Tout est fait avec NumPy uniquement : pas d'autodiff, pas de framework de
deep learning. C'est donc plus lent et moins optimisé qu'un vrai framework,
mais ça permet de comprendre exactement ce qui se passe "sous le capot".

================================================================================
"""

import numpy as np

# Pour la reproductibilité
np.random.seed(42)


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def softmax(x, axis=-1):
    """Softmax numériquement stable."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ==============================================================================
# OUTILS POUR COLLECTER LES PARAMÈTRES / GRADIENTS DE TOUT LE MODÈLE
# ==============================================================================
#
# Convention utilisée par chaque couche :
#   - Une couche "feuille" (Linear, LayerNorm, Embedding) expose :
#       .params_dict() -> dict {nom: tableau numpy (paramètre)}
#       .grads         -> dict {nom: tableau numpy (gradient)}, rempli par backward()
#   - Une couche "composite" (MultiHeadAttention, FeedForward, TransformerBlock,
#     GPT) expose :
#       .sublayers() -> liste de (nom, sous-couche)
#
# Les fonctions ci-dessous parcourent récursivement le modèle pour construire
# un dictionnaire global {nom_complet: tableau}. Comme les tableaux numpy sont
# mutables, l'optimiseur peut modifier les paramètres "in place" directement.

def collect_params(layer, prefix=""):
    params = {}
    if hasattr(layer, "params_dict"):
        for k, v in layer.params_dict().items():
            params[f"{prefix}{k}"] = v
    if hasattr(layer, "sublayers"):
        for name, sub in layer.sublayers():
            params.update(collect_params(sub, prefix=f"{prefix}{name}."))
    return params


def collect_grads(layer, prefix=""):
    grads = {}
    if hasattr(layer, "grads"):
        for k, v in layer.grads.items():
            grads[f"{prefix}{k}"] = v
    if hasattr(layer, "sublayers"):
        for name, sub in layer.sublayers():
            grads.update(collect_grads(sub, prefix=f"{prefix}{name}."))
    return grads


# ==============================================================================
# COUCHE LINEAIRE (Dense / Fully Connected)
# ==============================================================================

class Linear:
    """y = x @ W + b"""

    def __init__(self, in_dim, out_dim):
        # Initialisation "Xavier-like" pour stabiliser l'entraînement
        self.W = np.random.randn(in_dim, out_dim) * (1.0 / np.sqrt(in_dim))
        self.b = np.zeros(out_dim)
        self.grads = {}

    def forward(self, x):
        self.x = x  # on garde l'entrée en mémoire pour le backward
        return x @ self.W + self.b

    def backward(self, dout):
        # x: (..., in_dim), dout: (..., out_dim)
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        dout_flat = dout.reshape(-1, dout.shape[-1])

        self.grads["W"] = x_flat.T @ dout_flat
        self.grads["b"] = dout_flat.sum(axis=0)

        dx = dout @ self.W.T
        return dx

    def params_dict(self):
        return {"W": self.W, "b": self.b}


# ==============================================================================
# LAYER NORMALIZATION
# ==============================================================================

class LayerNorm:
    """Normalise sur la dernière dimension (d_model), puis applique gamma/beta."""

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self.grads = {}

    def forward(self, x):
        self.x = x
        self.mu = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.xhat = (x - self.mu) / self.std
        return self.gamma * self.xhat + self.beta

    def backward(self, dout):
        # Sommes sur toutes les dimensions sauf la dernière (features)
        sum_axes = tuple(range(dout.ndim - 1))
        self.grads["gamma"] = np.sum(dout * self.xhat, axis=sum_axes)
        self.grads["beta"] = np.sum(dout, axis=sum_axes)

        dxhat = dout * self.gamma

        # Formule classique de backprop de LayerNorm
        dx = (1.0 / self.std) * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True)
        )
        return dx

    def params_dict(self):
        return {"gamma": self.gamma, "beta": self.beta}


# ==============================================================================
# EMBEDDING (tokens + position)
# ==============================================================================

class Embedding:
    def __init__(self, vocab_size, d_model, max_len):
        self.tok_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(max_len, d_model) * 0.02
        self.grads = {}

    def forward(self, idx):
        # idx: (B, T) d'entiers (indices de tokens)
        self.idx = idx
        B, T = idx.shape
        self.B, self.T = B, T
        return self.tok_emb[idx] + self.pos_emb[:T]  # broadcast sur B

    def backward(self, dout):
        # dout: (B, T, d_model)
        dtok = np.zeros_like(self.tok_emb)
        T = self.T

        # On accumule le gradient pour chaque token utilisé (plusieurs
        # occurrences possibles -> on additionne, pas d'assignation simple)
        np.add.at(dtok, self.idx.reshape(-1), dout.reshape(-1, dout.shape[-1]))

        dpos_full = np.zeros_like(self.pos_emb)
        dpos_full[:T] = dout.sum(axis=0)  # somme sur le batch

        self.grads["tok_emb"] = dtok
        self.grads["pos_emb"] = dpos_full
        # Pas de dx à retourner : c'est la première couche du réseau

    def params_dict(self):
        return {"tok_emb": self.tok_emb, "pos_emb": self.pos_emb}


# ==============================================================================
# MULTI-HEAD SELF-ATTENTION (causale)
# ==============================================================================

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)

    def _split_heads(self, z, B, T):
        H, dh = self.n_heads, self.d_head
        return z.reshape(B, T, H, dh).transpose(0, 2, 1, 3)  # (B,H,T,dh)

    def _merge_heads(self, z, B, T):
        H, dh = self.n_heads, self.d_head
        return z.transpose(0, 2, 1, 3).reshape(B, T, H * dh)  # (B,T,D)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        Q = self.Wq.forward(x)
        K = self.Wk.forward(x)
        V = self.Wv.forward(x)

        Qh = self._split_heads(Q, B, T)
        Kh = self._split_heads(K, B, T)
        Vh = self._split_heads(V, B, T)

        # Scores d'attention : (B,H,T,T)
        scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(dh)

        # Masque causal : un token ne peut regarder que lui-même et le passé
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)

        attn = softmax(scores, axis=-1)  # (B,H,T,T)
        out = attn @ Vh  # (B,H,T,dh)
        out = self._merge_heads(out, B, T)  # (B,T,D)
        out = self.Wo.forward(out)

        # Cache nécessaire pour le backward
        self.cache = (Qh, Kh, Vh, attn, B, T, mask)
        return out

    def backward(self, dout):
        Qh, Kh, Vh, attn, B, T, mask = self.cache
        dh = self.d_head

        dout_o = self.Wo.backward(dout)  # (B,T,D)
        dout_o = self._split_heads(dout_o, B, T)  # (B,H,T,dh)

        # out = attn @ V
        dattn = dout_o @ Vh.transpose(0, 1, 3, 2)  # (B,H,T,T)
        dVh = attn.transpose(0, 1, 3, 2) @ dout_o  # (B,H,T,dh)

        # Backward du softmax : dscores = attn * (dattn - sum(dattn*attn))
        dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))
        dscores = np.where(mask, 0.0, dscores)
        dscores = dscores / np.sqrt(dh)

        # scores = Q @ K^T
        dQh = dscores @ Kh
        dKh = dscores.transpose(0, 1, 3, 2) @ Qh

        dQ = self._merge_heads(dQh, B, T)
        dK = self._merge_heads(dKh, B, T)
        dV = self._merge_heads(dVh, B, T)

        dx_q = self.Wq.backward(dQ)
        dx_k = self.Wk.backward(dK)
        dx_v = self.Wv.backward(dV)

        return dx_q + dx_k + dx_v  # même x utilisé pour Q, K, V -> on additionne

    def sublayers(self):
        return [("Wq", self.Wq), ("Wk", self.Wk), ("Wv", self.Wv), ("Wo", self.Wo)]


# ==============================================================================
# FEED-FORWARD (MLP avec ReLU)
# ==============================================================================

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, x):
        h = self.fc1.forward(x)
        self.relu_mask = (h > 0)
        a = h * self.relu_mask  # ReLU
        return self.fc2.forward(a)

    def backward(self, dout):
        da = self.fc2.backward(dout)
        dh = da * self.relu_mask
        return self.fc1.backward(dh)

    def sublayers(self):
        return [("fc1", self.fc1), ("fc2", self.fc2)]


# ==============================================================================
# BLOC TRANSFORMER (Pre-LN, comme GPT-2)
#
#   x ---> LN1 ---> Attention ---> (+résiduel) ---> LN2 ---> FFN ---> (+résiduel)
#   |__________________________________|         |_______________________|
# ==============================================================================

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        a = self.attn.forward(self.ln1.forward(x))
        x1 = x + a  # résiduel 1

        f = self.ffn.forward(self.ln2.forward(x1))
        x2 = x1 + f  # résiduel 2

        return x2

    def backward(self, dout):
        # x2 = x1 + f  =>  dx1 reçoit dout (résiduel) + gradient venant de f
        df = dout
        dx1_from_ffn = self.ln2.backward(self.ffn.backward(df))
        dx1 = dout + dx1_from_ffn

        # x1 = x + a  =>  dx reçoit dx1 (résiduel) + gradient venant de a
        da = dx1
        dx_from_attn = self.ln1.backward(self.attn.backward(da))
        dx = dx1 + dx_from_attn

        return dx

    def sublayers(self):
        return [("ln1", self.ln1), ("attn", self.attn), ("ln2", self.ln2), ("ffn", self.ffn)]


# ==============================================================================
# MODELE COMPLET : GPT-like (decoder-only Transformer)
# ==============================================================================

class GPT:
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len):
        self.embed = Embedding(vocab_size, d_model, max_len)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)

    def forward(self, idx):
        x = self.embed.forward(idx)
        for blk in self.blocks:
            x = blk.forward(x)
        x = self.ln_f.forward(x)
        logits = self.head.forward(x)
        return logits

    def backward(self, dlogits):
        dx = self.head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        self.embed.backward(dx)

    def sublayers(self):
        layers = [("embed", self.embed)]
        for i, blk in enumerate(self.blocks):
            layers.append((f"block{i}", blk))
        layers.append(("ln_f", self.ln_f))
        layers.append(("head", self.head))
        return layers


# ==============================================================================
# FONCTION DE PERTE : Softmax + Cross-Entropy (avec gradient analytique)
# ==============================================================================

def cross_entropy_loss(logits, targets):
    """
    logits: (B, T, V)
    targets: (B, T) entiers (indices des tokens corrects)
    Retourne (loss scalaire, dlogits)
    """
    B, T, V = logits.shape
    probs = softmax(logits, axis=-1)

    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]

    correct_probs = probs[b_idx, t_idx, targets]
    loss = -np.mean(np.log(correct_probs + 1e-12))

    # Gradient du softmax+cross-entropy : probs - one_hot(target), divisé par B*T
    dlogits = probs.copy()
    dlogits[b_idx, t_idx, targets] -= 1
    dlogits /= (B * T)

    return loss, dlogits


# ==============================================================================
# OPTIMISEUR ADAM (implémenté à la main)
# ==============================================================================

class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k, p in params.items():
            g = grads.get(k)
            if g is None:
                continue
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)

            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)

            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)  # mise à jour "in place"


# ==============================================================================
# DONNEES D'ENTRAINEMENT : modèle de langage au niveau caractère
# ==============================================================================

# Remplace ce texte par n'importe quel corpus (plus c'est gros, mieux c'est,
# mais l'entraînement sera plus lent en pur NumPy).
TEXT = (
    "le transformer est une architecture de reseau de neurones "
    "qui utilise un mecanisme d'attention pour traiter des sequences. "
    "il a ete introduit dans larticle attention is all you need en 2017. "
    "les modeles de langage modernes comme gpt sont bases sur cette architecture. "
) * 4

chars = sorted(list(set(TEXT)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

data = np.array([stoi[c] for c in TEXT], dtype=np.int64)


def get_batch(data, batch_size, block_size):
    max_start = len(data) - block_size - 1
    idxs = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in idxs])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in idxs])
    return x, y


# ==============================================================================
# GENERATION DE TEXTE (sampling autoregressif)
# ==============================================================================

def generate(model, start_str, length, block_size, temperature=0.8):
    idx = [stoi.get(c, 0) for c in start_str]
    idx = list(idx)

    for _ in range(length):
        context = idx[-block_size:]
        # padding si trop court
        if len(context) < block_size:
            context = [0] * (block_size - len(context)) + context

        x = np.array([context])  # (1, block_size)
        logits = model.forward(x)  # (1, block_size, vocab_size)
        last_logits = logits[0, -1] / temperature
        probs = softmax(last_logits)

        next_id = np.random.choice(len(probs), p=probs)
        idx.append(int(next_id))

    return "".join(itos[i] for i in idx)


# ==============================================================================
# ENTRAINEMENT
# ==============================================================================

def train():
    # Hyperparamètres (volontairement petits pour rester rapide en pur NumPy)
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 4 * d_model
    block_size = 32  # longueur de contexte
    batch_size = 16
    n_steps = 500
    lr = 3e-3
    grad_clip = 1.0

    print(f"Taille du vocabulaire : {vocab_size}")

    model = GPT(vocab_size, d_model, n_heads, n_layers, d_ff, max_len=block_size)
    params = collect_params(model)

    n_params = sum(p.size for p in params.values())
    print(f"Nombre total de paramètres : {n_params:,}")

    optimizer = Adam(params, lr=lr)

    for step in range(1, n_steps + 1):
        x, y = get_batch(data, batch_size, block_size)

        logits = model.forward(x)
        loss, dlogits = cross_entropy_loss(logits, y)
        model.backward(dlogits)

        grads = collect_grads(model)

        # Gradient clipping global (norme L2) pour la stabilité
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > grad_clip:
            scale = grad_clip / (total_norm + 1e-6)
            for g in grads.values():
                g *= scale

        optimizer.step(params, grads)

        if step % 50 == 0 or step == 1:
            print(f"step {step:4d} | loss = {loss:.4f}")

    print("\n--- Génération de texte après entraînement ---")
    sample = generate(model, start_str="le transformer", length=200,
                       block_size=block_size, temperature=0.7)
    print(sample)

    return model


if __name__ == "__main__":
    train()
