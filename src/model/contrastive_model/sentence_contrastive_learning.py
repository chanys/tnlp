import logging

import torch
from torch.utils.data import DataLoader

from src.data.data_utils import read_bioasq_examples_from_file

from transformers import set_seed

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceTransformerContrastiveLearning(object):
    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def test(self):
        ds = read_bioasq_examples_from_file(self.config.data.test_jsonl)
        examples = []
        for i in range(len(ds)):
            example = ds[i]
            examples.append(InputExample(texts=[example["query"], example["chosen"], example["rejected"]]))

        model = SentenceTransformer(self.config.model.model_id)

        test_evaluator = TripletEvaluator.from_input_examples(examples)
        test_evaluator(model)

    def train(self):
        ds_train = read_bioasq_examples_from_file(self.config.data.train_jsonl)
        train_examples = []
        for i in range(len(ds_train)):
            example = ds_train[i]
            train_examples.append(InputExample(texts=[example["query"], example["chosen"], example["rejected"]]))
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.config.hyperparams.batch_size)

        ds_validation = read_bioasq_examples_from_file(self.config.data.validation_jsonl)
        validation_examples = []
        for i in range(len(ds_validation)):
            example = ds_validation[i]
            validation_examples.append(InputExample(texts=[example["query"], example["chosen"], example["rejected"]]))

        model = SentenceTransformer(self.config.model.model_id)
        train_loss = losses.TripletLoss(model=model)

        evaluator = TripletEvaluator.from_input_examples(validation_examples)

        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=self.config.hyperparams.epoch,
                  evaluator=evaluator, output_path=self.config.processing.output_dir)
