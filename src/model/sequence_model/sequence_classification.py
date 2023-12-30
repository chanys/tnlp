import os
import logging

import torch
import numpy as np

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer, AutoConfig,
    set_seed
)

from src.data.data_utils import read_sequence_examples_from_file
from src.model.metric import compute_f1_metric, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceClassification(object):
    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def predict_batch(self, batch, tokenizer, model):
        inputs = tokenizer(
            batch["text"], padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        d = {}
        with torch.no_grad():
            output = model(**inputs)
            logits = output.logits
            pred_label = torch.argmax(logits, axis=-1)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            d["predicted_label"] = pred_label.cpu().numpy()
            d["probabilities"] = probabilities.cpu().numpy()

            if "label" in batch:
                loss = torch.nn.functional.cross_entropy(
                    logits, batch["label"].to(self.device), reduction="none"
                )
                d["loss"] = loss.cpu().numpy()

        return d

    def inference(self):
        ds = read_sequence_examples_from_file(self.config.data.inference_jsonl)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.model_id  # path to the saved model
        ).to(self.device)

        num_examples = len(ds)
        num_batches = (
                              num_examples + self.config.hyperparams.batch_size - 1
                      ) // self.config.hyperparams.batch_size

        all_results = []

        for batch_index in tqdm(range(num_batches), desc="Processing batches"):
            start_index = batch_index * self.config.hyperparams.batch_size
            end_index = min((batch_index + 1) * self.config.hyperparams.batch_size, num_examples)

            batch = ds[start_index:end_index]

            result_batch = self.predict_batch(batch, tokenizer, model)
            all_results.append(result_batch)

        # Combine results from all batches
        combined_results = {}
        for key in all_results[0].keys():
            combined_results[key] = torch.cat(
                [torch.tensor(result[key]) for result in all_results]
            )

        id2label_mapping = AutoConfig.from_pretrained(self.config.model.model_id).id2label
        predicted_label_strings = [id2label_mapping[label_index.item()] for label_index in
                                   combined_results["predicted_label"]]

        return {"text": ds["text"], "label": predicted_label_strings}

    def test(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model.model_id).to(self.device)

        ds_test = read_sequence_examples_from_file(self.config.data.test_jsonl)

        index2label = {index: label for index, label in enumerate(ds_test.features["label"].names)}

        ds_test = ds_test.map(
            lambda x: tokenizer(x["text"], padding=True, truncation=True), batched=True
        )

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=self.config.processing.output_dir,
                per_device_eval_batch_size=self.config.hyperparams.batch_size,
                disable_tqdm=False,
            ),
            # compute_metrics=lambda p: compute_f1_metric(p, index2label),
            compute_metrics=lambda p: compute_f1_metric(p.predictions.argmax(-1), p.label_ids, index2label),
            tokenizer=tokenizer,
        )

        prediction_output = trainer.predict(ds_test)

        # print F1 score
        # f1_metrics = compute_f1_metric(prediction_output, index2label)
        f1_metrics = compute_f1_metric(prediction_output.predictions.argmax(-1), prediction_output.label_ids,
                                       index2label)
        print(f"F1 Metrics on Test Set: {f1_metrics}")

        # print confusion matrix
        y_preds = np.argmax(prediction_output.predictions, axis=1)
        y_valid = np.array(ds_test["label"])
        labels = ds_test.features["label"].names
        confusion_matrix(labels, y_valid, y_preds)

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)

        ds_train = read_sequence_examples_from_file(self.config.data.train_jsonl)

        index2label = {index: label for index, label in enumerate(ds_train.features["label"].names)}
        label2index = {label: index for index, label in enumerate(ds_train.features["label"].names)}

        ds_train = ds_train.map(
            lambda x: tokenizer(x["text"], padding=True, truncation=True), batched=True
        )

        ds_validation = read_sequence_examples_from_file(self.config.data.validation_jsonl)
        ds_validation = ds_validation.map(
            lambda x: tokenizer(x["text"], padding=True, truncation=True), batched=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.model_id, num_labels=len(index2label), id2label=index2label, label2id=label2index,
        ).to(self.device)

        logging_steps = len(ds_train) // self.config.hyperparams.batch_size
        training_args = TrainingArguments(
            output_dir=self.config.processing.output_dir,
            num_train_epochs=self.config.hyperparams.epoch,
            learning_rate=self.config.hyperparams.learning_rate,
            per_device_train_batch_size=self.config.hyperparams.batch_size,
            per_device_eval_batch_size=self.config.hyperparams.batch_size,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="steps",
            save_steps=self.config.hyperparams.save_steps,
            save_total_limit=self.config.hyperparams.save_total_limit,
            disable_tqdm=False,
            logging_steps=logging_steps,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            # compute_metrics=lambda p: compute_f1_metric(p, index2label),
            compute_metrics=lambda p: compute_f1_metric(p.predictions.argmax(-1), p.label_ids, index2label),
            train_dataset=ds_train,
            eval_dataset=ds_validation,
            tokenizer=tokenizer,
        )
        trainer.train()

        model.save_pretrained(
            os.path.join(self.config.processing.output_dir, "final_model")
        )
        tokenizer.save_pretrained(
            os.path.join(self.config.processing.output_dir, "final_model")
        )
