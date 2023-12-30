import os
import logging
from typing import Dict, List

import torch
import numpy as np

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer, AutoConfig, DataCollatorForTokenClassification, AutoModelForTokenClassification,
    set_seed
)

from src.data.data_utils import read_token_examples_from_file
from src.model.metric import compute_seqeval_metric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_and_align_labels(examples: Dict[str, List], tokenizer) -> Dict[str, List]:
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for index, label in enumerate(examples[f"labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=index)

        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None or word_id == previous_word_id:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
            previous_word_id = word_id
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


class TokenClassification(object):
    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def predict_batch(self, batch, tokenizer, model):
        inputs = tokenizer(
            batch["tokens"],
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
        )

        # let's keep track of the word_ids
        batch_word_ids = []
        for batch_index in range(len(batch["tokens"])):
            word_ids = np.array(inputs.word_ids(batch_index=batch_index))
            attention_mask = inputs.attention_mask[batch_index]
            batch_word_ids.append(word_ids[attention_mask == 1])

        # after this, type(inputs) will go from BatchEncoder to a Python Dict, and inputs.word_ids will not be available
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
            logits = output.logits
            pred_label_ids = torch.argmax(logits, dim=-1).cpu().numpy()

        batch_labels = []
        for batch_index in range(len(batch["tokens"])):
            word_ids = batch_word_ids[batch_index]
            subword_label_ids = pred_label_ids[batch_index]

            previous_word_id = None
            label_ids = []
            for index, word_id in enumerate(word_ids):
                if word_id is None or word_id == previous_word_id:  # only take the prediction from each first sub-word
                    pass
                else:
                    label_ids.append(subword_label_ids[index])
                previous_word_id = word_id
            batch_labels.append(label_ids)

        return {"tokens": batch["tokens"], "labels": batch_labels}

    def inference(self):
        ds = read_token_examples_from_file(self.config.data.inference_jsonl)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)
        model = AutoModelForTokenClassification.from_pretrained(
            self.config.model.model_id  # path to the saved model
        ).to(self.device)

        num_examples = len(ds)
        num_batches = (
                              num_examples + self.config.hyperparams.batch_size - 1
                      ) // self.config.hyperparams.batch_size

        id2label_mapping = AutoConfig.from_pretrained(self.config.model.model_id).id2label

        all_tokens = []
        all_label_strings = []
        for batch_index in tqdm(range(num_batches), desc="Processing batches"):
            start_index = batch_index * self.config.hyperparams.batch_size
            end_index = min((batch_index + 1) * self.config.hyperparams.batch_size, num_examples)

            batch = ds[start_index:end_index]
            result_batch = self.predict_batch(batch, tokenizer, model)

            all_tokens.extend(result_batch["tokens"])
            all_label_strings.extend(
                [[id2label_mapping[label_id] for label_id in label_ids] for label_ids in result_batch["labels"]])

        return {"tokens": all_tokens, "labels": all_label_strings}

    def test(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)
        model = AutoModelForTokenClassification.from_pretrained(self.config.model.model_id).to(self.device)

        ds_test = read_token_examples_from_file(self.config.data.test_jsonl)
        ds_test_tokenized = ds_test.map(
            tokenize_and_align_labels,
            batched=True,
            batch_size=self.config.hyperparams.batch_size,
            fn_kwargs={"tokenizer": tokenizer},
        )

        index2label = {index: label for index, label in enumerate(ds_test.features["labels"].feature.names)}

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=self.config.processing.output_dir,
                per_device_eval_batch_size=self.config.hyperparams.batch_size,
                disable_tqdm=False,
            ),
            compute_metrics=lambda pred: compute_seqeval_metric(pred, index2label),
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        prediction_output = trainer.predict(ds_test_tokenized)

        # print F1 score
        seqeval_metrics = compute_seqeval_metric((prediction_output.predictions, prediction_output.label_ids),
                                                 index2label)
        print(f"Seqeval Metrics on Test Set: {seqeval_metrics}")

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)

        ds_train = read_token_examples_from_file(self.config.data.train_jsonl)

        index2label = {index: label for index, label in enumerate(ds_train.features["labels"].feature.names)}
        label2index = {label: index for index, label in enumerate(ds_train.features["labels"].feature.names)}

        ds_train_tokenized = ds_train.map(
            tokenize_and_align_labels,
            batched=True,
            batch_size=self.config.hyperparams.batch_size,
            fn_kwargs={"tokenizer": tokenizer},
        )

        ds_validation = read_token_examples_from_file(self.config.data.validation_jsonl)

        ds_validation_tokenized = ds_validation.map(
            tokenize_and_align_labels,
            batched=True,
            batch_size=self.config.hyperparams.batch_size,
            fn_kwargs={"tokenizer": tokenizer},
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        model = AutoModelForTokenClassification.from_pretrained(
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
            train_dataset=ds_train_tokenized,
            eval_dataset=ds_validation_tokenized,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda pred: compute_seqeval_metric(pred, index2label)
        )
        trainer.train()

        model.save_pretrained(
            os.path.join(self.config.processing.output_dir, "final_model")
        )
        tokenizer.save_pretrained(
            os.path.join(self.config.processing.output_dir, "final_model")
        )
