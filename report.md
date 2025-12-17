# POS Tagging with RoBERTa Report

## Overview

I implemented a neural part-of-speech tagger using XLM-RoBERTa and PyTorch. The model classifies each token in German sentences into one of 18 Universal POS tags.

---

## Implementation

### Data

I used the German Universal Dependencies corpus (de_gsd):
- Training set: 13,814 sentences
- Validation set: 799 sentences
- Test set: 977 sentences

### Components

**1. Data Loading & Tokenization** (`src/data.py`)

I loaded the dataset from Hugging Face as parquet files and tokenized it using XLM-RoBERTa's tokenizer. The tokenization handles subword alignment by assigning POS labels only to the first subword of each word, with subsequent subwords receiving -100 (ignored during training).

**2. Model Definition** (`src/model.py`)

I implemented a simple architecture:
- XLM-RoBERTa encoder (768-dimensional outputs, frozen)
- Linear projection layer (768 → 18)

The encoder parameters are frozen to avoid expensive optimization of 250M parameters. Only the linear layer is trained.

**3. Training Loop** (`src/train.py`)

I used CrossEntropyLoss with `ignore_index=-100` to skip padding tokens, and Adam optimizer with learning rate 1e-3. The training loop tracks both training and validation loss per epoch.

**4. Evaluation** (`src/evaluate.py`)

I computed per-token accuracy on the validation set, excluding tokens with label -100 (padding and non-first subwords).

---

## Results

**Training Configuration**:
- Epochs: 3
- Batch size: 16
- Learning rate: 1e-3
- Optimizer: Adam

**Performance**:
- Validation accuracy: 93.68%

The learning curve shows both training and validation loss decreasing across epochs, indicating the model is learning properly.

---

## Discussion

### Comparison to HMM

The HMM-based tagger achieved 15-19% accuracy, while this RoBERTa-based approach achieves 93.68%. The difference is significant and comes from several factors:

1. **Context**: The HMM only considers 1-2 previous tags. XLM-RoBERTa uses self-attention to consider the entire sentence context in both directions.

2. **Unknown words**: The HMM used a naive uniform distribution for unknown words. XLM-RoBERTa handles this through subword tokenization - even unseen words are broken into known subword pieces.

3. **Pretrained knowledge**: XLM-RoBERTa was pretrained on massive multilingual corpora. This gives it strong linguistic representations before we even start training on POS data. The HMM had no such prior knowledge.

4. **Representation**: The HMM treats words as discrete symbols. XLM-RoBERTa represents words as 768-dimensional vectors that capture semantic and syntactic information.

---
