import os
import logging

import torch
from tqdm import tqdm
import numpy as np

from src.data.data_utils import read_sequence_examples_from_file
from src.data.span_dataset import SpanDataset
from src.model.sequence_model.custom_sequence_model import CustomSequenceModel
from src.model.model_utils import prepare_optimizer_and_scheduler

from transformers import set_seed, AutoTokenizer, AutoConfig
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomSequenceClassification(object):
    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def _create_model(self, mode, num_classes=None):
        assert mode in {"train", "test", "inference"}

        if mode == "test" or mode == "inference":
            # get the number of classes from the saved model directory
            auto_config = AutoConfig.from_pretrained(self.config.model.model_path)
            num_classes = auto_config.num_labels
            logger.info(f"num_classes={num_classes}")

            model = CustomSequenceModel(self.config, num_classes)
            model_path = os.path.join(self.config.model.model_path, 'checkpoint.pth.tar')
            logger.info('Loading model from {}'.format(model_path))

            checkpoint = torch.load(model_path)

            model.load_state_dict(checkpoint['state_dict'])
            return model.to(self.device)
        else:  # train from scratch
            logger.info('Creating model from scratch')
            assert num_classes is not None
            model = CustomSequenceModel(self.config, num_classes)
            return model.to(self.device)

    def prepare_dataset(self, data: Dataset, tokenizer, shuffle=False):
        dataset = SpanDataset(data, tokenizer, self.config.hyperparams.batch_size,
                              self.config.hyperparams.max_seq_length, shuffle=shuffle)
        dataset.encode()
        return dataset.data_loader()

    def inference(self, test_examples):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)

        dataloader = self.prepare_dataset(test_examples, tokenizer, shuffle=False)

        model = self._create_model("inference")
        model.eval()
        input_fields = tokenizer.model_input_names

        predictions = []
        predictions_probs = []
        softmax_probs = []
        for batch in dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k in input_fields}
            preds, argmax_probs, out_labels, loss, softmax = self.predict(model, inputs['input_ids'],
                                                                          inputs['attention_mask'])
            predictions.extend(preds)
            predictions_probs.extend(argmax_probs)
            softmax_probs.extend(softmax)

        return predictions, predictions_probs, softmax_probs

    def predict(self, model, input_ids, attention_masks, labels=None):
        with torch.no_grad():
            outputs = model.forward(input_ids, attention_masks, labels)

        if labels is not None:
            loss = model.label_criteria(outputs.logits, labels.to(self.device))
            loss = loss.cpu().numpy()
        else:
            loss = None

        # we can't directly convert any tensor requiring gradients to numpy arrays.
        # so we need to call .detach() first to remove the computational graph tracking.
        # .cpu is in case the tensor is on the GPU, in which case you need to move it back to the CPU to convert it to a tensor

        logits_cpu = outputs.logits.detach().cpu()
        softmax = torch.softmax(logits_cpu, dim=-1)  # do a softmax over the prediction probabilities
        argmax_probs = torch.max(softmax, dim=-1).values.numpy()  # probability of the argmax

        preds = logits_cpu.numpy()  # the raw logit scores
        preds = np.argmax(preds, axis=-1)  # label index of the argmax

        if labels is not None:
            out_labels = labels.detach().cpu().numpy()
        else:
            out_labels = None

        return preds, argmax_probs, out_labels, loss, softmax.numpy()

    def test(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)

        ds_test = read_sequence_examples_from_file(self.config.data.test_jsonl)
        test_dataloader = self.prepare_dataset(ds_test, tokenizer, shuffle=False)

        model = self._create_model("test")
        model.eval()

        input_fields = tokenizer.model_input_names + ["label"]
        predictions = []
        gold_labels = []
        losses = []
        for batch in test_dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k in input_fields}
            preds, argmax_probs, out_labels, loss, softmax = self.predict(model, inputs['input_ids'],
                                                                          inputs['attention_mask'],
                                                                          labels=inputs['label'])
            predictions.extend(preds)
            gold_labels.extend(out_labels)
            losses.append(loss)

        predictions = np.asarray(predictions)
        gold_labels = np.asarray(gold_labels)

        mean_loss = sum(losses) / len(losses)
        print('Test loss is %.5f' % mean_loss)
        print('Test accuracy is %.2f' % (float(np.sum(predictions == gold_labels)) / len(predictions)))

    def train(self):
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)  # can specify: cache_dir=<local_dir>

        # training data
        ds_train = read_sequence_examples_from_file(self.config.data.train_jsonl)
        train_dataloader = self.prepare_dataset(ds_train, tokenizer, shuffle=True)
        num_classes = len(ds_train.features["label"].names)

        # validation data
        ds_validation = read_sequence_examples_from_file(self.config.data.validation_jsonl)
        validation_dataloader = self.prepare_dataset(ds_validation, tokenizer, shuffle=False)

        # model
        model = self._create_model("train", num_classes=num_classes)

        num_training_steps = (
                                         len(train_dataloader) * self.config.hyperparams.epoch) / self.config.hyperparams.batch_size

        optimizer, scheduler = prepare_optimizer_and_scheduler(model, self.config.hyperparams.learning_rate,
                                                               self.config.hyperparams.warmup_steps, num_training_steps)

        logger.info("***** Running training *****")
        logger.info("  Number of examples = %d", len(ds_train))
        logger.info("  Number of Epochs = %d", self.config.hyperparams.epoch)
        logger.info("  Training batch size = %d", self.config.hyperparams.batch_size)
        logger.info("  Number of training steps = %d", num_training_steps)

        input_fields = tokenizer.model_input_names + ["label"]
        for epoch_counter in range(self.config.hyperparams.epoch):
            model.train()  # sets module to training mode

            losses = []
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for batch_index, batch in enumerate(epoch_iterator):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k in input_fields}
                optimizer.zero_grad()
                outputs = model(inputs['input_ids'], inputs['attention_mask'], labels=inputs['label'])
                loss = model.label_criteria(outputs.logits, inputs['label'].to(self.device))
                losses.append(loss)
                loss.backward()  # calculate and accumulate the gradients

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()  # nudge the parameters in the opposite direction of the gradient, in order to decrease the loss

            scheduler.step()  # Update learning rate schedule

            mean_loss = sum(losses) / len(losses)
            print('Training loss at epoch %d is %.5f' % (epoch_counter, mean_loss))
            # wandb.log({'training loss': mean_loss})     # wandb logging

            model.eval()
            predictions = []
            gold_labels = []
            losses = []
            for batch in validation_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k in input_fields}
                preds, argmax_probs, out_labels, loss, softmax = self.predict(model, inputs['input_ids'],
                                                                              inputs['attention_mask'],
                                                                              labels=inputs['label'])
                predictions.extend(preds)
                gold_labels.extend(out_labels)
                losses.append(loss)

            mean_loss = sum(losses) / len(losses)
            print('Validation loss at epoch %d is %.5f' % (epoch_counter, mean_loss))

            predictions = np.asarray(predictions)
            gold_labels = np.asarray(gold_labels)
            val_accuracy = float(np.sum(predictions == gold_labels)) / len(predictions)
            print('Validation accuracy at epoch %d is %.2f' % (epoch_counter, val_accuracy))
            # wandb.log({'validation loss': mean_loss})           # wandb logging
            # wandb.log({'validation accuracy': val_accuracy})    # wandb logging

        model.save_model(self.config.processing.output_dir, optimizer)
        tokenizer.save_pretrained(self.config.processing.output_dir)
