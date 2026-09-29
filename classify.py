"""Predict a label with a classification-fine-tuned GPT-like checkpoint."""

import argparse
from pathlib import Path

import torch

from finetuner import DEFAULT_CLASSIFICATION_SAVE_PATH, GPTClassifier
from tokenizer import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Classifier avec un checkpoint GPT fine-tuné.")
    parser.add_argument("--model", type=Path, default=DEFAULT_CLASSIFICATION_SAVE_PATH)
    parser.add_argument("--text", required=True, help="Texte à classer.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError("Checkpoint de classification introuvable : {}".format(args.model))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA a été demandé mais n'est pas disponible.")

    checkpoint = torch.load(args.model, map_location=device)
    if checkpoint.get("task") != "classification":
        raise ValueError("Le checkpoint fourni n'est pas un checkpoint de classification.")
    labels = checkpoint.get("labels")
    config = checkpoint.get("config")
    if not labels or not config or "model_state" not in checkpoint:
        raise ValueError("Le checkpoint de classification est incomplet.")

    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(args.text.strip())[:config["context_length"]]
    if not token_ids:
        raise ValueError("Le texte doit contenir au moins un token.")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)

    model = GPTClassifier(config, len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(input_ids, lengths), dim=1)[0]
    label_id = probabilities.argmax().item()
    print("label: {} | confiance: {:.1%}".format(labels[label_id], probabilities[label_id].item()))


if __name__ == "__main__":
    main()