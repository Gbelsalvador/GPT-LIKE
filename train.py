
import torch
from loss import calc_loss_batch, calc_loss_loader, evaluate_model
from generate import generate_and_print_sample, generate_text_simple
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen , global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch , target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # calculate loss gradients
            optimizer.step() #update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step +=1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"epoch {epoch + 1} | step {global_step:06d} | "
                      f"train loss {train_loss:.4f} | val loss {val_loss:.4f}")


        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.to("cpu")
# device = "cpu"
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# num_epochs = 10
# train_losses , val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq=5, eval_iter=5,
#     start_context="je suis fatiguer", tokenizer=tokenizer)
