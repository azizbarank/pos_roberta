import torch
from src.data import load_corpus, get_tokenizer, tokenize_and_align_labels, create_dataloaders
from src.model import POSTagger
from src.train import train
from src.evaluate import evaluate


def main():
    dataset = load_corpus()
    tokenizer = get_tokenizer()

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    train_loader, val_loader, test_loader = create_dataloaders(tokenized_dataset, tokenizer)

    model = POSTagger()

    model, train_losses, val_losses = train(model, train_loader, val_loader, num_epochs=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = evaluate(model, val_loader, device)
    print(f"Validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
