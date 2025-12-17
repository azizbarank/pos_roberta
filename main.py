from src.data import load_corpus, get_tokenizer, tokenize_and_align_labels, create_dataloaders
from src.model import POSTagger


def main():
    print("Loading dataset...")
    dataset = load_corpus()
    print(f"Train: {len(dataset['train'])} examples")
    print(f"Validation: {len(dataset['validation'])} examples")
    print(f"Test: {len(dataset['test'])} examples")

    print("\nSample sentence:")
    sample = dataset["train"][0]
    print(f"Tokens: {sample['tokens']}")
    print(f"POS tags: {sample['upos']}")

    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()

    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(tokenized_dataset, tokenizer)

    print("\nSample batch:")
    batch = next(iter(train_loader))
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels shape: {batch['labels'].shape}")

    print("\nLoading model...")
    model = POSTagger()

    print("\nTesting forward pass...")
    logits = model(batch["input_ids"], batch["attention_mask"])
    print(f"logits shape: {logits.shape}")


if __name__ == "__main__":
    main()
