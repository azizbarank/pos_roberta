import torch.nn as nn
from transformers import XLMRobertaModel


class POSTagger(nn.Module):
    def __init__(self, num_labels=18):
        super().__init__()
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.classifier = nn.Linear(768, num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)
        return logits
