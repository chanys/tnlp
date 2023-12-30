import os
import logging

import torch
import torch.nn as nn
from transformers import AutoModel

from src.model.linears import Linears

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomSpanPairModel(nn.Module):
    def __init__(self, configuration, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(configuration.model.model_id, output_hidden_states=True)
        self.encoder_dropout = nn.Dropout(p=configuration.hyperparams.encoder_dropout)

        self.label_criteria = torch.nn.CrossEntropyLoss()

        # The first layer is encoder_dim * 4 because we are concatenating:
        # - span 1 representation (span1)
        # - span 2 representation (span2)
        # - span1 * span2
        # - abs(span1 - span2)
        self.classification_fcl = Linears(
            [self.encoder.config.hidden_size * 4, configuration.hyperparams.fcl_hidden_dim, num_classes],
            dropout_prob=configuration.hyperparams.linear_dropout
        )

    def forward(self, batch):
        all_encoder_outputs = self.encoder(batch["input_ids"], attention_mask=batch["attention_mask"])
        encoder_outputs = all_encoder_outputs[0]
        encoder_outputs = self.encoder_dropout(encoder_outputs)

        head_embeddings = []
        tail_embeddings = []
        for batch_index in range(batch["input_ids"].shape[0]):
            start = batch["head_start"][batch_index]
            end = batch["head_end"][batch_index]
            head_span_embeddings = encoder_outputs[batch_index, start:end + 1, :]
            head_embeddings.append(torch.mean(head_span_embeddings, dim=0))

            start = batch["tail_start"][batch_index]
            end = batch["tail_end"][batch_index]
            tail_span_embeddings = encoder_outputs[batch_index, start:end + 1, :]
            tail_embeddings.append(torch.mean(tail_span_embeddings, dim=0))

        m1_reprs = torch.stack(head_embeddings, dim=0)
        m2_reprs = torch.stack(tail_embeddings, dim=0)

        m_reprs = torch.cat(
            [m1_reprs, m2_reprs, m1_reprs * m2_reprs, torch.abs(m1_reprs - m2_reprs)],
            dim=1
        )

        return self.classification_fcl(m_reprs)

    def predict(self, batch):
        classification_logits = self.forward(batch)
        label_type_probs, label_type_predictions = torch.max(torch.softmax(classification_logits, dim=1), dim=1)

        return label_type_predictions, label_type_probs, classification_logits

    def save_model(self, model_dir, optimizer, index2label):
        print("=> Saving model to {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': optimizer.state_dict()}
        checkpoint = {"state_dict": self.state_dict(), "num_classes": self.num_classes, "index2label": index2label}
        torch.save(checkpoint, os.path.join(model_dir, "checkpoint.pth.tar"))

        self.encoder.config.save_pretrained(model_dir)
