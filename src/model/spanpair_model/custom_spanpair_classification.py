import logging
import os
from typing import Dict, List
from collections import defaultdict

import numpy as np
import torch

from tqdm import tqdm

from transformers import set_seed, AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader

from src.model.model_utils import get_tokenizer
from src.model.metric import compute_f1_metric
from src.data.data_utils import read_nyth_relation_examples_from_file
from src.model.spanpair_model.custom_spanpair_model import CustomSpanPairModel

logger = logging.getLogger(__name__)


def tokenize(examples: Dict[str, List], tokenizer, max_seq_length) -> Dict[str, List]:
    tokenized_inputs = tokenizer(examples["tokens"], padding="max_length", truncation=True, is_split_into_words=True,
                                 max_length=max_seq_length)

    head_start = []
    head_end = []
    tail_start = []
    tail_end = []
    use_instance = []
    for batch_index in range(len(examples["tokens"])):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)

        head_token_start = examples["head_token_start"][batch_index]
        head_token_end = examples["head_token_end"][batch_index]
        tail_token_start = examples["tail_token_start"][batch_index]
        tail_token_end = examples["tail_token_end"][batch_index]

        head_subword_start = head_subword_end = tail_subword_start = tail_subword_end = None
        for i, word_id in enumerate(word_ids):
            if word_id == head_token_start and head_subword_start is None:
                head_subword_start = i
            if word_id == head_token_end:
                head_subword_end = i
            if word_id == tail_token_start and tail_subword_start is None:
                tail_subword_start = i
            if word_id == tail_token_end:
                tail_subword_end = i

        head_start.append(head_subword_start)
        head_end.append(head_subword_end)
        tail_start.append(tail_subword_start)
        tail_end.append(tail_subword_end)

        if head_subword_start is None or head_subword_end is None or tail_subword_start is None or tail_subword_end is None:
            use_instance.append(False)
        else:
            use_instance.append(True)

    tokenized_inputs["head_start"] = head_start
    tokenized_inputs["head_end"] = head_end
    tokenized_inputs["tail_start"] = tail_start
    tokenized_inputs["tail_end"] = tail_end
    tokenized_inputs["use_instance"] = use_instance

    return tokenized_inputs


class CustomSpanPairClassification(object):
    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def convert_batch_to_model_inputs(self, batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch])
        attention_mask = torch.tensor([item["attention_mask"] for item in batch])
        label = torch.tensor([item["label"] for item in batch])
        head_start = [item["head_start"] for item in batch]
        head_end = [item["head_end"] for item in batch]
        tail_start = [item["tail_start"] for item in batch]
        tail_end = [item["tail_end"] for item in batch]

        return {"input_ids": input_ids.to(self.device), "attention_mask": attention_mask.to(self.device),
                "label": label.to(self.device), "head_start": head_start, "head_end": head_end,
                "tail_start": tail_start, "tail_end": tail_end}

    def score_examples(self, model, dataloader, index2label, epoch_counter=None):
        model.eval()
        predictions = []
        gold_labels = []
        losses = []
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=False):
                model_batch = self.convert_batch_to_model_inputs(batch)
                batch_predictions, batch_probabilities, batch_logits = model.predict(model_batch)

                predictions.extend(batch_predictions.tolist())
                gold_labels.extend([item["label"] for item in batch])

                loss = model.label_criteria(batch_logits, model_batch["label"])
                losses.append(loss)

        mean_loss = sum(losses) / len(losses)
        if epoch_counter is not None:
            print('Validation loss at epoch %d is %.5f' % (epoch_counter, mean_loss))
        else:
            print('Validation loss is %.5f' % mean_loss)

        predictions = np.asarray(predictions)
        gold_labels = np.asarray(gold_labels)
        val_accuracy = float(np.sum(predictions == gold_labels)) / len(predictions)
        if epoch_counter is not None:
            print('Validation accuracy at epoch %d is %.2f' % (epoch_counter, val_accuracy))
        else:
            print('Validation accuracy is %.2f' % val_accuracy)

        f1_scores = compute_f1_metric(np.array(predictions), np.array(gold_labels), index2label)
        print(f1_scores)

    def test(self):
        tokenizer = get_tokenizer(self.config.model.tokenizer_id, max_length=self.config.model.max_seq_length)

        ds = read_nyth_relation_examples_from_file(self.config.data.test_jsonl)

        ds_tokenized = ds.map(tokenize, batched=True, batch_size=None,
                              fn_kwargs={"tokenizer": tokenizer, "max_seq_length": self.config.model.max_seq_length})
        ds_tokenized = ds_tokenized.filter(lambda example: example["use_instance"] == True)

        dataloader = DataLoader(ds_tokenized, batch_size=self.config.hyperparams.batch_size, shuffle=True,
                                collate_fn=lambda x: x)

        model, index2label = self._create_model("test")
        print(f"index2label={index2label}")

        self.score_examples(model, dataloader, index2label)

    def train(self):
        tokenizer = get_tokenizer(self.config.model.tokenizer_id, max_length=self.config.model.max_seq_length)

        ds_train = read_nyth_relation_examples_from_file(self.config.data.train_jsonl)
        ds_train_tokenized = ds_train.map(tokenize, batched=True, batch_size=None, fn_kwargs={"tokenizer": tokenizer,
                                                                                              "max_seq_length": self.config.model.max_seq_length})
        logger.info(f"len(ds_train_tokenized)={len(ds_train_tokenized)}")

        ds_train_tokenized = ds_train_tokenized.filter(lambda example: example["use_instance"] == True)

        logger.info(f"after filtering for overly long texts: len(ds_train_tokenized)={len(ds_train_tokenized)}")

        train_dataloader = DataLoader(ds_train_tokenized, batch_size=self.config.hyperparams.batch_size, shuffle=True,
                                      collate_fn=lambda x: x)

        label_counts = defaultdict(int)
        for example in ds_train_tokenized:
            label_counts[example['label']] += 1
        print(f"label_counts={label_counts}")

        ds_validation = read_nyth_relation_examples_from_file(self.config.data.validation_jsonl)
        ds_validation_tokenized = ds_validation.map(tokenize, batched=True, batch_size=None,
                                                    fn_kwargs={"tokenizer": tokenizer,
                                                               "max_seq_length": self.config.model.max_seq_length})

        logger.info(f"len(ds_validation_tokenized)={len(ds_validation_tokenized)}")
        ds_validation_tokenized = ds_validation_tokenized.filter(lambda example: example["use_instance"] == True)
        logger.info(
            f"after filtering for overly long texts: len(ds_validation_tokenized)={len(ds_validation_tokenized)}")

        validation_dataloader = DataLoader(ds_validation_tokenized, batch_size=self.config.hyperparams.batch_size,
                                           shuffle=True, collate_fn=lambda x: x)

        index2label = {index: label for index, label in enumerate(ds_train.features["label"].names)}

        num_classes = len(set(ds_train["label"]))
        assert num_classes is not None

        model, _ = self._create_model("train", num_classes=num_classes)

        num_training_steps = (
                                         len(train_dataloader) * self.config.hyperparams.epoch) / self.config.hyperparams.batch_size

        optimizer = AdamW(params=model.parameters(), lr=self.config.hyperparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.hyperparams.warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info("***** Running training *****")
        logger.info("  Number of examples = %d", len(ds_train))
        logger.info("  Number of Epochs = %d", self.config.hyperparams.epoch)
        logger.info("  Training batch size = %d", self.config.hyperparams.batch_size)
        logger.info("  Number of training steps = %d", num_training_steps)

        for epoch_counter in range(self.config.hyperparams.epoch):
            model.train()
            print('Epoch {}, lr {}'.format(epoch_counter, optimizer.param_groups[0]['lr']))

            losses = []
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for batch_index, batch in enumerate(epoch_iterator):
                # inputs = {k: v.to(self.device) for k, v in batch.items() if k in input_fields}
                optimizer.zero_grad()
                model_batch = self.convert_batch_to_model_inputs(batch)
                outputs = model(model_batch)

                loss = model.label_criteria(outputs, model_batch["label"])
                # loss = loss * (1 / self.config.hyperparams.gradient_accumulation_steps)
                losses.append(loss)
                loss.backward()  # calculate and accumulate the gradients

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()  # nudge the parameters in the opposite direction of the gradient, in order to decrease the loss

            scheduler.step()  # Update learning rate schedule

            mean_loss = sum(losses) / len(losses)
            print('Training loss at epoch %d is %.5f' % (epoch_counter, mean_loss))

            #### check performance on dev_examples
            self.score_examples(model, validation_dataloader, index2label, epoch_counter=epoch_counter)

        model.save_model(self.config.processing.output_dir, optimizer, index2label)
        tokenizer.save_pretrained(self.config.processing.output_dir)

    def _create_model(self, mode, num_classes=None):
        assert mode in {"train", "test", "inference"}

        if mode == "test" or mode == "inference":
            # get the number of classes from the saved model directory
            # auto_config = AutoConfig.from_pretrained(self.config.model.model_path)
            # num_classes = auto_config.num_labels

            model_path = os.path.join(self.config.model.model_path, 'checkpoint.pth.tar')
            logger.info('Loading model from {}'.format(model_path))
            checkpoint = torch.load(model_path)

            num_classes = checkpoint.get('num_classes')
            print(f"num_classes={num_classes}")

            index2label = checkpoint.get("index2label")

            model = CustomSpanPairModel(self.config, num_classes)
            model.load_state_dict(checkpoint['state_dict'])

            return model.to(self.device), index2label
        else:  # train from scratch
            logger.info('Creating model from scratch')
            assert num_classes is not None
            model = CustomSpanPairModel(self.config, num_classes)
            return model.to(self.device), None
