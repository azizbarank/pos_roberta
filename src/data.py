from datasets import load_dataset
from transformers import XLMRobertaTokenizerFast, DataCollatorForTokenClassification
from torch.utils.data import DataLoader


def load_corpus():
    base_url = "https://huggingface.co/datasets/universal-dependencies/universal_dependencies/resolve/refs/convert/parquet/de_gsd/"
    data_files = {
        "train": base_url + "train/0000.parquet",
        "validation": base_url + "validation/0000.parquet",
        "test": base_url + "test/0000.parquet",
    }
    dataset = load_dataset("parquet", data_files=data_files)
    return dataset


def get_tokenizer():
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    return tokenizer


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=False, skip_index=-100):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=True,
    )

    labels = []
    for i, label in enumerate(examples["upos"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(skip_index)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else skip_index)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def create_dataloaders(tokenized_dataset, tokenizer, batch_size=16):
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    val_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        tokenized_dataset["test"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_dataloader, val_dataloader, test_dataloader
