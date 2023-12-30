import os
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.triplet_dataset import TripletDataset
from src.model.contrastive_model.custom_triplet_contrastive_model import CustomTripletContrastiveModel
from src.data.data_utils import read_bioasq_as_triplet_examples_from_file

from src.model.model_utils import get_tokenizer

from transformers import set_seed, AdamW, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomTripletContrastiveLearning(object):
    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def test(self):
        # TODO This will be very similar to the score_examples() method
        pass

    def train(self):
        tokenizer = get_tokenizer(self.config.model.tokenizer_id, max_length=self.config.model.max_seq_length)

        ds_train = read_bioasq_as_triplet_examples_from_file(self.config.data.train_jsonl)
        logger.info("Triplet examples loaded")
        ds_train = TripletDataset(ds_train, self.config)
        logger.info("ds_train.numberize")
        ds_train.numberize(tokenizer)
        logger.info("ds_train.numberize ended")
        train_dataloader = DataLoader(ds_train, batch_size=self.config.hyperparams.batch_size, shuffle=True,
                                      collate_fn=ds_train.collate_fn)

        ds_validation = read_bioasq_as_triplet_examples_from_file(self.config.data.validation_jsonl)
        ds_validation = TripletDataset(ds_validation, self.config)
        ds_validation.numberize(tokenizer)
        validation_dataloader = DataLoader(ds_validation, batch_size=self.config.hyperparams.batch_size, shuffle=False,
                                           collate_fn=ds_validation.collate_fn)

        model = self._create_model("train")

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
                optimizer.zero_grad()
                chosen_distance, rejected_distance = model(batch)
                loss = model.calculate_loss(chosen_distance, rejected_distance)
                losses.append(loss)
                loss.backward()  # calculate and accumulate the gradients

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()  # nudge the parameters in the opposite direction of the gradient, in order to decrease the loss

            scheduler.step()  # Update learning rate schedule

            mean_loss = sum(losses) / len(losses)
            print('Training loss at epoch %d is %.5f' % (epoch_counter, mean_loss))

            self.score_examples(model, validation_dataloader, epoch_counter=epoch_counter)

        model.save_model(self.config.processing.output_dir, optimizer)
        tokenizer.save_pretrained(self.config.processing.output_dir)

    def score_examples(self, model, dataloader, epoch_counter=None):
        model.eval()

        correct_count = 0
        total_count = 0
        losses = []
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=False):
                chosen_distance, rejected_distance = model(batch)
                loss = model.calculate_loss(chosen_distance, rejected_distance)
                losses.append(loss)

                chosen_distance = chosen_distance.cpu().numpy()
                rejected_distance = rejected_distance.cpu().numpy()

                correct_count += sum(
                    chosen_d < rejected_d for chosen_d, rejected_d in zip(chosen_distance, rejected_distance))
                total_count += len(chosen_distance)

        mean_loss = sum(losses) / len(losses)
        if epoch_counter is not None:
            print('Validation loss at epoch %d is %.5f' % (epoch_counter, mean_loss))
        else:
            print('Validation loss is %.5f' % mean_loss)

        val_accuracy = float(correct_count) / total_count
        if epoch_counter is not None:
            print('Validation accuracy at epoch %d is %.2f' % (epoch_counter, val_accuracy))
        else:
            print('Validation accuracy is %.2f' % val_accuracy)

    def _create_model(self, mode, num_classes=2):
        assert mode in {"train", "test", "inference"}

        if mode == "test" or mode == "inference":
            model_path = os.path.join(self.config.model.model_path, 'checkpoint.pth.tar')
            logger.info('Loading model from {}'.format(model_path))
            checkpoint = torch.load(model_path)

            model = CustomTripletContrastiveModel(self.config, num_classes)
            model.load_state_dict(checkpoint['state_dict'])
            return model.to(self.device)
        else:  # train from scratch
            logger.info('Creating model from scratch')
            model = CustomTripletContrastiveModel(self.config, num_classes)
            return model.to(self.device)
