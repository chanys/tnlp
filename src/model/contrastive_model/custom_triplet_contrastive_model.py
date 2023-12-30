import os

import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F


class FCLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class TripletDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return 1 - F.cosine_similarity(x, y)


class CustomTripletContrastiveModel(nn.Module):
    def __init__(self, configuration, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(configuration.model.model_id, output_hidden_states=True)

        self.label_criteria = torch.nn.CrossEntropyLoss()

        self.anchor_fcl = FCLayer(self.encoder.config.hidden_size)
        self.candidate_fcl = FCLayer(self.encoder.config.hidden_size)

        self.distance_metric = TripletDistance()
        self.triplet_margin = 0.1

    def forward(self, batch):
        anchor_encoded = self.encoder(batch.anchor_input_ids, attention_mask=batch.anchor_attention_mask)
        anchor_encoded = anchor_encoded[0]  # last hidden state

        chosen_encoded = self.encoder(batch.chosen_input_ids, attention_mask=batch.chosen_attention_mask)
        chosen_encoded = chosen_encoded[0]

        rejected_encoded = self.encoder(batch.rejected_input_ids, attention_mask=batch.rejected_attention_mask)
        rejected_encoded = rejected_encoded[0]

        assert len(batch.anchor_input_ids) == len(batch.chosen_input_ids) == len(batch.rejected_input_ids)

        anchor_embeddings = []
        for batch_index in range(len(batch.anchor_input_ids)):
            start = batch.anchor_start[batch_index]
            end = batch.anchor_end[batch_index]
            anchor_span_embeddings = anchor_encoded[batch_index, start:end + 1, :]
            anchor_embeddings.append(torch.mean(anchor_span_embeddings, dim=0))
        anchor_embeddings = torch.stack(anchor_embeddings, dim=0)
        anchor_output = self.anchor_fcl(anchor_embeddings).squeeze(dim=1)

        chosen_embeddings = [chosen_encoded[i, 0: 1, :] for i in range(len(batch.chosen_input_ids))]  # [CLS]
        chosen_embeddings = torch.stack(chosen_embeddings, dim=0)
        chosen_output = self.candidate_fcl(chosen_embeddings).squeeze(dim=1)

        rejected_embeddings = [rejected_encoded[i, 0: 1, :] for i in range(len(batch.rejected_input_ids))]  # [CLS]
        rejected_embeddings = torch.stack(rejected_embeddings, dim=0)
        rejected_output = self.candidate_fcl(rejected_embeddings).squeeze(dim=1)

        chosen_distance = self.distance_metric(anchor_output, chosen_output)
        rejected_distance = self.distance_metric(anchor_output, rejected_output)

        return chosen_distance, rejected_distance

    def calculate_loss(self, chosen_distance, rejected_distance):
        losses = F.relu(chosen_distance - rejected_distance + self.triplet_margin)
        return losses.mean()

    def predict(self, batch):
        classification_logits = self.forward(batch)
        label_type_probs, label_type_predictions = torch.max(torch.softmax(classification_logits, dim=1), dim=1)

        return label_type_predictions, label_type_probs, classification_logits

    def save_model(self, model_dir, optimizer):
        print("=> Saving model to {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': optimizer.state_dict()}
        checkpoint = {"state_dict": self.state_dict()}
        torch.save(checkpoint, os.path.join(model_dir, "checkpoint.pth.tar"))

        self.encoder.config.save_pretrained(model_dir)
