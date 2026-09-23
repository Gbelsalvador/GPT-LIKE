"""
================================================================================
ARCHITECTURE GPT-LIKE 
================================================================================

  - Embedding de tokens + embedding positionnel
  - Multi-Head Self-Attention causale
  - Feed-Forward (MLP) avec activation GELU
  - LayerNorm
  - Connexions résiduelles (pre-LN, comme GPT-2)
  - Tête de sortie (projection vers le vocabulaire)
  - Entraînement avec optimiseur Adam + autodiff (backprop automatique)
  - Génération de texte (sampling) après entraînement
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# MULTI-HEAD SELF-ATTENTION (causale)
# ==============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        Q = self.Wq(x).view(B, T, H, dh).transpose(1, 2)  # (B,H,T,dh)
        K = self.Wk(x).view(B, T, H, dh).transpose(1, 2)
        V = self.Wv(x).view(B, T, H, dh).transpose(1, 2)

        # Scores d'attention : (B,H,T,T)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)

        # Masque causal : un token ne peut regarder que lui-même et le passé
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V  # (B,H,T,dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
        return self.Wo(out)


# ==============================================================================
# FEED-FORWARD (MLP avec GELU)
# ==============================================================================

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# BLOC TRANSFORMER (Pre-LN, comme GPT-2)
#
#   x ---> LN1 ---> Attention ---> (+résiduel) ---> LN2 ---> FFN ---> (+résiduel)
# ==============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # résiduel 1
        x = x + self.ffn(self.ln2(x))    # résiduel 2
        return x


# ==============================================================================
# MODELE COMPLET : GPT-like (decoder-only Transformer)
# ==============================================================================

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Initialisation des poids (style GPT-2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1,T)

        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B,T,D)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """idx: (B,T) tokens de contexte. Retourne (B, T+max_new_tokens)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]  # tronque au contexte max
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, next_id], dim=1)

        self.train()
        return idx


# ==============================================================================
# DONNEES D'ENTRAINEMENT : modèle de langage au niveau caractère
# ==============================================================================

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

device = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long, device=device)


def get_batch(data, batch_size, block_size):
    max_start = len(data) - block_size - 1
    idxs = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in idxs])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in idxs])
    return x, y


def decode(ids):
    return "".join(itos[i] for i in ids)


# ==============================================================================
# ENTRAINEMENT
# ==============================================================================

def train():
    torch.manual_seed(42)

    # Hyperparamètres
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 4 * d_model
    block_size = 32
    batch_size = 16
    n_steps = 500
    lr = 3e-3

    print(f"Device : {device}")
    print(f"Taille du vocabulaire : {vocab_size}")

    model = GPT(vocab_size, d_model, n_heads, n_layers, d_ff, max_len=block_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre total de paramètres : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(1, n_steps + 1):
        x, y = get_batch(data, batch_size, block_size)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"step {step:4d} | loss = {loss.item():.4f}")

    print("\n--- Génération de texte après entraînement ---")
    context = torch.tensor([[stoi[c] for c in "le transformer"]], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=200, temperature=0.7)
    print(decode(out[0].tolist()))

    return model


if __name__ == "__main__":
    train()
