import torch
import tiktoken


def get_tokenizer():
    """Return the GPT-2 tokenizer used by the training pipeline."""
    return tiktoken.get_encoding("gpt2")


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
