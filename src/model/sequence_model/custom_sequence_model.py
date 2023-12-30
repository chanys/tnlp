import os
import logging

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomSequenceModel(nn.Module):
    def __init__(self, configuration, num_classes=None):
        super().__init__()

        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            configuration.model.model_id, num_labels=num_classes
        )

        self.label_criteria = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_masks, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        # If we are not using a sequence classification head, we might do the following:
        # outputs = self.encoder(input_ids, attention_masks)
        # cls = outputs.last_hidden_state[:, 0]
        # logits = self.linear(cls)
        return outputs

    def save_model(self, model_dir, optimizer):
        print('=> Saving model to {}'.format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': optimizer.state_dict()}
        checkpoint = {'state_dict': self.state_dict()}
        torch.save(checkpoint, os.path.join(model_dir, 'checkpoint.pth.tar'))

        self.encoder.config.save_pretrained(model_dir)
