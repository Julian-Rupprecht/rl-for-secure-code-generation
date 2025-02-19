import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, model, config):
        super(Model, self).__init__()
        self.model = model
        self.config = config
        #self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels=None): 
        outputs = self.model(input_ids, attention_mask=attention_mask)[0]

        #outputs = self.dropout(outputs)
        
        probs = torch.sigmoid(outputs)
        if labels is not None: 
            labels = labels.unsqueeze(1).float()
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs, labels)
            return loss, probs
        else: 
            return probs