import torch
import torch.nn as nn


def train(model, train_dataloader, val_dataloader, num_epochs=3, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_outputs = batch["labels"].to(device)

            predicted_logits = model(inputs, attention_mask)

            loss = loss_function(
                predicted_logits.view(-1, predicted_logits.shape[-1]),
                gold_outputs.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_dataloader)
        train_losses.append(train_loss)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                gold_outputs = batch["labels"].to(device)

                predicted_logits = model(inputs, attention_mask)
                loss = loss_function(
                    predicted_logits.view(-1, predicted_logits.shape[-1]),
                    gold_outputs.view(-1)
                )
                val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)
        val_losses.append(val_loss)

    return model, train_losses, val_losses
