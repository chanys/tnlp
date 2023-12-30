import torch
from peft import get_peft_model
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.model.model_utils import print_trainable_parameters, get_lora_config


class RagEnd2EndModel(torch.nn.Module):
    def __init__(self, configuration) -> None:
        super(RagEnd2EndModel, self).__init__()

        self.retriever_model = AutoModel.from_pretrained(configuration.model.retriever_model_id, device_map={"": 0})
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(configuration.model.retriever_tokenizer_id)

        self.generator_model = AutoModelForCausalLM.from_pretrained(configuration.model.generator_model_id, load_in_8bit=True, device_map="auto")
        self.generator_tokenizer = AutoTokenizer.from_pretrained(configuration.model.generator_tokenizer_id)
        self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        self.generator_tokenizer.add_eos_token = True

        self.generator_model = get_peft_model(self.generator_model, peft_config=get_lora_config(configuration))
        print_trainable_parameters(self.generator_model)

    def save_models_tokenizers(self, retriever_output_dir, generator_output_dir):
        self.retriever_model.save_pretrained(retriever_output_dir)
        self.retriever_tokenizer.save_pretrained(retriever_output_dir)

        self.generator_model.save_pretrained(generator_output_dir)
        self.generator_tokenizer.save_pretrained(generator_output_dir)

    def _retrieval_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.retriever_model(input_ids, attention_mask)[0]  # shape=(batch_size, max_seq_len, hidden-dim)
        embeddings = self._mean_pooling(token_embeddings, attention_mask)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)  # shape=(batch_size, hidden-dim)

    def _generator_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        gen_outputs = self.generator_model(input_ids=input_ids, attention_mask=attention_mask)  # CausalLMOutputWithPast
        return gen_outputs.logits  # shape=(batch_size, generator_max_seq_len, vocab_size)

    def forward(self, task: str, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if task == "retrieval":
            return self._retrieval_forward(input_ids, attention_mask)
        else:
            return self._generator_forward(input_ids, attention_mask)

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        model_output.shape=(batch_size, max_seq_length, hidden_dim)
        attention_mask.shape=(batch_size, max_seq_length)
        """
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()  # shape=token_embeddings.shape
        # sum embeddings along the subword dimension, then divide by number of subwords
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
