import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = output.loss
        # Logits shape: [batch_size, max_length, vocab_size], e.g. [8, 512, 32100]
        preds = torch.argmax(output.logits, dim=-1)
        return loss, preds 