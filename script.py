"""Train and sample from the GPT-like model.

Example:
	python script.py --epochs 5 --prompt "je suis"
"""

import argparse
import random
from pathlib import Path

import torch

from config import GPT_CONFIG_SMALL
from dataset import create_dataloaders
from generate import generate_and_print_sample
from GPT import GPTModel
from tokenizer import get_tokenizer
from train import train_model_simple

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "train.txt"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.pt"


def parse_args():
	parser = argparse.ArgumentParser(description="Entraîner un petit GPT-like.")
	parser.add_argument(
		"--text", type=Path, default=DEFAULT_DATA_PATH,
		help="Fichier texte à utiliser comme corpus (défaut: data/train.txt).",
	)
	parser.add_argument("--prompt", default="je suis", help="Texte de départ.")
	parser.add_argument("--generate", action="store_true", help="Générer avec un modèle déjà enregistré.")
	parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
		help="Checkpoint à charger pour la génération (défaut: model.pt).")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=2)
	parser.add_argument("--block-size", type=int, default=32)
	parser.add_argument("--stride", type=int, default=16)
	parser.add_argument("--eval-freq", type=int, default=10)
	parser.add_argument("--eval-iter", type=int, default=5)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
	parser.add_argument("--save", type=Path, help="Chemin de sauvegarde du modèle.")
	return parser.parse_args()


def main():
	args = parse_args()
	random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
	if device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA a été demandé mais n'est pas disponible.")

	tokenizer = get_tokenizer()
	if args.generate:
		if not args.model.is_file():
			raise FileNotFoundError("Modèle introuvable : {}".format(args.model))
		checkpoint = torch.load(args.model, map_location=device)
		model = GPTModel(checkpoint["config"]).to(device)
		model.load_state_dict(checkpoint["model_state"])
		print("model loaded from:", args.model)
		print("sample:")
		generate_and_print_sample(model, tokenizer, device, args.prompt)
		return

	if not args.text.is_file():
		raise FileNotFoundError("Corpus introuvable : {}".format(args.text))
	text = args.text.read_text(encoding="utf-8")
	train_loader, val_loader = create_dataloaders(
		text, tokenizer, args.block_size, args.stride, args.batch_size
	)

	config = dict(GPT_CONFIG_SMALL)
	config["context_length"] = max(config["context_length"], args.block_size)
	model = GPTModel(config).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

	print("device:", device)
	print("parameters:", sum(parameter.numel() for parameter in model.parameters()))
	train_model_simple(
		model, train_loader, val_loader, optimizer, device, args.epochs,
		args.eval_freq, args.eval_iter, args.prompt, tokenizer
	)
	print("\nsample:")
	generate_and_print_sample(model, tokenizer, device, args.prompt)

	if args.save:
		torch.save({"model_state": model.state_dict(), "config": config}, args.save)
		print("model saved to:", args.save)


if __name__ == "__main__":
	main()
