import torch
from torch.utils.data import DataLoader, Dataset

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, max(0, len(token_ids) - max_length), stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length +1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# Backward-compatible name for notebooks that used the original spelling.
GPTDatsetV1 = GPTDataset


DEFAULT_TEXT = """
je suis fatiguer de coder.
j'ai besoin d'argent.
j'ai peur de femme claire c'est pour cela que je fais l'informatique.
je suis tellement con que je ne sais plus comment resoudre le probleme.
je code tous le jours mais j'ai pas d'argent.
donc jusqu'aujourdhui j'ai pas ecouter de marcher pour la data science.
arreter d'etudier mes amis la vie est dure.
je te considere comme un frere.
apres avoir ecouter ça j'ai arreté d'ecouter des chansons romantique.
si seulement tu savais à quel point faire un LLM c'est dur.
laisse petit si j'arrive à faire un agent ia c'est finis il va braqué pour moi.
je t'aime comme un ami a fait d'elle son enemie.
la vie est dur quand on a pour hobbit d'entrainer de modele d'ia.
je suis fatiguer de coder.
j'ai besoin d'argent.
j'ai peur de femme claire c'est pour cela que je fais l'informatique.
je suis tellement con que je ne sais plus comment resoudre le probleme.
je code tous le jours mais j'ai pas d'argent.
donc jusqu'aujourdhui j'ai pas ecouter de marcher pour la data science.
arreter d'etudier mes amis la vie est dure.
je te considere comme un frere.
apres avoir ecouter ça j'ai arreté d'ecouter des chansons romantique.
si seulement tu savais à quel point faire un LLM c'est dur.
laisse petit si j'arrive à faire un agent ia c'est finis il va braqué pour moi.
je t'aime comme un ami a fait d'elle son enemie.
la vie est dur quand on a pour hobbit d'entrainer de modele d'ia.
je suis fatiguer de coder.
j'ai besoin d'argent.
j'ai peur de femme claire c'est pour cela que je fais l'informatique.
je suis tellement con que je ne sais plus comment resoudre le probleme.
je code tous le jours mais j'ai pas d'argent.
donc jusqu'aujourdhui j'ai pas ecouter de marcher pour la data science.
arreter d'etudier mes amis la vie est dure.
je te considere comme un frere.
apres avoir ecouter ça j'ai arreté d'ecouter des chansons romantique.
si seulement tu savais à quel point faire un LLM c'est dur.
laisse petit si j'arrive à faire un agent ia c'est finis il va braqué pour moi.
je t'aime comme un ami a fait d'elle son enemie.
la vie est dur quand on a pour hobbit d'entrainer de modele d'ia.
"""


def create_dataloaders(text, tokenizer, max_length=32, stride=16, batch_size=2,
                       train_ratio=0.9):
    """Split text and return train/validation DataLoaders."""
    split = int(train_ratio * len(text))
    train_dataset = GPTDataset(text[:split], tokenizer, max_length, stride)
    val_dataset = GPTDataset(text[split:], tokenizer, max_length, stride)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Le texte est trop court pour max_length={}.".format(max_length))

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
    )
    