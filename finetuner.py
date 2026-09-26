"""Supervised fine-tuning for the GPT-like model using JSONL instruction data.

Example:
    python finetuner.py --model model.pt --data data/instructions.jsonl
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from GPT import GPTModel
from tokenizer import get_tokenizer

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "instructions.jsonl"
DEFAULT_MODEL_PATH = PROJECT_DIR / "model.pt"
DEFAULT_SAVE_PATH = PROJECT_DIR / "finetuned_model.pt"
IGNORE_INDEX = -100


class InstructionDataset(Dataset):
    """Tokenized instruction/response pairs with loss masked on the prompt."""

    def __init__(self, data_path, tokenizer, context_length):
        self.examples = []
        max_sequence_length = context_length + 1

        with data_path.open("r", encoding="utf-8") as data_file:
            for line_number, line in enumerate(data_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "JSON invalide à la ligne {} de {}.".format(line_number, data_path)
                    ) from error

                instruction = record.get("instruction")
                response = record.get("response")
                extra_input = record.get("input", "")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError("'instruction' doit être une chaîne non vide (ligne {}).".format(line_number))
                if not isinstance(response, str) or not response.strip():
                    raise ValueError("'response' doit être une chaîne non vide (ligne {}).".format(line_number))
                if not isinstance(extra_input, str):
                    raise ValueError("'input' doit être une chaîne (ligne {}).".format(line_number))

                prompt = "### Instruction:\n{}".format(instruction.strip())
                if extra_input.strip():
                    prompt += "\n\n### Contexte:\n{}".format(extra_input.strip())
                prompt += "\n\n### Réponse:\n"

                prompt_ids = tokenizer.encode(prompt)
                response_ids = tokenizer.encode(response.strip())
                end_token = tokenizer.eot_token
                if len(response_ids) >= max_sequence_length:
                    response_ids = response_ids[:max_sequence_length - 1] + [end_token]
                else:
                    response_ids.append(end_token)

                # Keep the end of long prompts and the start/end of the response.
                prompt_budget = max_sequence_length - len(response_ids)
                prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
                sequence = prompt_ids + response_ids
                if len(sequence) < 2:
                    raise ValueError("Exemple trop court à la ligne {}.".format(line_number))

                input_ids = sequence[:-1]
                labels = sequence[1:]
                prompt_target_count = max(0, len(prompt_ids) - 1)
                labels[:prompt_target_count] = [IGNORE_INDEX] * prompt_target_count
                self.examples.append((input_ids, labels))

        if not self.examples:
            raise ValueError("Aucun exemple d'entraînement trouvé dans {}.".format(data_path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_examples(batch):
    max_length = max(len(input_ids) for input_ids, _ in batch)
    input_batch = torch.zeros((len(batch), max_length), dtype=torch.long)
    label_batch = torch.full((len(batch), max_length), IGNORE_INDEX, dtype=torch.long)
    for row, (input_ids, labels) in enumerate(batch):
        input_batch[row, :len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
        label_batch[row, :len(labels)] = torch.tensor(labels, dtype=torch.long)
    return input_batch, label_batch


def calc_batch_loss(model, input_batch, label_batch, device):
    input_batch = input_batch.to(device)
    label_batch = label_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), label_batch.flatten(), ignore_index=IGNORE_INDEX
    )


def average_loss(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_batch, label_batch in data_loader:
            total_loss += calc_batch_loss(model, input_batch, label_batch, device).item()
    model.train()
    return total_loss / len(data_loader)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuner supervisé du modèle GPT-like.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                        help="Dataset JSONL (défaut: data/instructions.jsonl).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Checkpoint pré-entraîné à charger (défaut: model.pt).")
    parser.add_argument("--save", type=Path, default=DEFAULT_SAVE_PATH,
                        help="Checkpoint fine-tuné à écrire (défaut: finetuned_model.pt).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs et --batch-size doivent être supérieurs à zéro.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate doit être supérieur à zéro.")
    if not args.data.is_file():
        raise FileNotFoundError("Dataset introuvable : {}".format(args.data))
    if not args.model.is_file():
        raise FileNotFoundError("Checkpoint pré-entraîné introuvable : {}".format(args.model))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA a été demandé mais n'est pas disponible.")

    checkpoint = torch.load(args.model, map_location=device)
    if "config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("Le checkpoint doit contenir 'config' et 'model_state'.")
    config = checkpoint["config"]
    tokenizer = get_tokenizer()
    dataset = InstructionDataset(args.data, tokenizer, config["context_length"])

    if len(dataset) < 2:
        raise ValueError("Il faut au moins deux exemples pour séparer entraînement et validation.")
    validation_size = max(1, int(len(dataset) * 0.1))
    training_size = len(dataset) - validation_size
    train_dataset, val_dataset = random_split(
        dataset, (training_size, validation_size),
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_examples
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_examples
    )

    model = GPTModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    print("device:", device)
    print("examples:", len(dataset), "| train:", training_size, "| validation:", validation_size)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for input_batch, label_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_batch_loss(model, input_batch, label_batch, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = average_loss(model, val_loader, device)
        print("epoch {}/{} | train loss {:.4f} | val loss {:.4f}".format(
            epoch + 1, args.epochs, train_loss, val_loss
        ))

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, args.save)
    print("fine-tuned model saved to:", args.save)


if __name__ == "__main__":
    main()
