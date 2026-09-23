import torch
from tokenizer import token_ids_to_text, text_to_token_ids

def generate_text_simple(model, idx, max_new_tokens, context_size):
    model_device = next(model.parameters()).device
    idx = idx.to(model_device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(next(model.parameters()).device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,idx=encoded, max_new_tokens=50, context_size=context_size
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", ""))
    model.train()
    