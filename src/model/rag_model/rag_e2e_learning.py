import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, AdamW, get_linear_schedule_with_warmup

from src.data.rag_dataset import RagDataset
from src.data.data_utils import read_rag_examples_from_file
from src.model.rag_model.rag_e2e_model import RagEnd2EndModel
from src.model.rag_model.rag_e2e_utils import (
    calculate_generator_loss,
    get_scaled_dot_product, calculate_cross_entropy_loss
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RagEnd2End(object):
    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def test(self):
        pass

    def train(self):
        model = self._create_model("train")

        ds_train = read_rag_examples_from_file(self.config.data.train_jsonl)

        ds_train = RagDataset(ds_train, self.config, model.retriever_tokenizer, model.generator_tokenizer)
        ds_train.numberize()
        train_dataloader = DataLoader(ds_train, batch_size=self.config.hyperparams.batch_size, shuffle=True,
                                      collate_fn=ds_train.collate_fn)

        num_training_steps = (len(train_dataloader) * self.config.hyperparams.epoch) / (
                self.config.hyperparams.batch_size)

        optimizer = AdamW(params=model.parameters(), lr=self.config.hyperparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.hyperparams.warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info("***** Running training *****")
        logger.info("  Number of examples = %d", len(train_dataloader))
        logger.info("  Number of Epochs = %d", self.config.hyperparams.epoch)
        logger.info("  Training batch size = %d", self.config.hyperparams.batch_size)
        logger.info("  Number of training steps = %d", num_training_steps)

        for epoch_counter in range(self.config.hyperparams.epoch):
            model.train()
            optimizer.zero_grad()

            losses = []
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for batch_index, batch in enumerate(epoch_iterator):
                query_embeddings = model("retrieval", batch.query_input_ids, batch.query_attention_mask)  # shape=(batch_size, hidden_dim)
                passage_embeddings = model("retrieval", batch.passage_input_ids, batch.passage_attention_mask)  # shape=(batch_size, hidden_dim)

                # Calculate cosine similarity between each query and passage. Since we have done L2-norm on the
                # query embeddings and passage embeddings, this is now simply a dot-product
                cosine_logits = get_scaled_dot_product(query_embeddings, passage_embeddings, self.config.hyperparams.temperature)

                # pass in similarity between (query, passage), i.e. for each query, its similarity to all batch passages
                loss_query = calculate_cross_entropy_loss(cosine_logits)

                # when we do cosine_logits.t(), we get similarity between (passage, query)
                loss_passage = calculate_cross_entropy_loss(cosine_logits.t())
                retriever_loss = (loss_query + loss_passage) / 2.0

                generator_logits = model("generation", batch.qpa_input_ids, batch.qpa_attention_mask)
                generator_loss = calculate_generator_loss(generator_logits, batch.qpa_input_ids, batch.qpa_attention_mask)

                combined_loss = retriever_loss + generator_loss
                combined_loss.backward()
                losses.append(combined_loss.item())

                if (batch_index + 1) % self.config.hyperparams.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            mean_loss = sum(losses) / len(losses)
            print('Training loss at epoch %d is %.5f' % (epoch_counter, mean_loss))

        model.save_models_tokenizers(self.config.processing.retriever_output_dir, self.config.processing.generator_output_dir)

    def _create_model(self, mode):
        assert mode in {"train", "test", "inference"}

        if mode == "test" or mode == "inference":
            pass
        else:           # train from scratch
            logger.info('Creating model from scratch')
            model = RagEnd2EndModel(self.config)
            return model.to(self.device)
