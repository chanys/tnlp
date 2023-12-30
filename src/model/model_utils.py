import logging
import random
from typing import Dict
import os

import numpy as np
import torch

from transformers import AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, AutoTokenizer, \
    AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftModel, PeftConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


weekdays = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
weekends = {"Saturday", "Sunday"}

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
""" The above is in Jinja template, which can be used like so: tokenizer.apply_chat_template(chat, tokenize=False) 
The above template in more readable format:
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ '<|user|>\n' + message['content'] + eos_token }}
    {% elif message['role'] == 'system' %}
        {{ '<|system|>\n' + message['content'] + eos_token }}
    {% elif message['role'] == 'assistant' %}
        {{ '<|assistant|>\n'  + message['content'] + eos_token }}
    {% endif %}
    {% if loop.last and add_generation_prompt %}
        {{ '<|assistant|>' }}
    {% endif %}
{% endfor %}

For more information, REF https://huggingface.co/docs/transformers/main/en/chat_templating
"""

def print_tokenizer_information(tokenizer):
    logger.info("Tokenizer Vocab Size: %s", tokenizer.vocab_size)
    logger.info("Tokenizer Model Max Length: %s", tokenizer.model_max_length)
    logger.info("Tokenizer Model Input Names: %s", tokenizer.model_input_names)

    tokens2ids = list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))
    logger.info("Special tokens to IDs: %s", tokens2ids)
    # [('[UNK]', 100), ('[SEP]', 102), ('[PAD]', 0), ('[CLS]', 101), ('[MASK]', 103)]

    text = "Tokenizing text is a core task of NLP"
    encoded_text = tokenizer(text)
    logger.info("Encoded Text: %s", encoded_text)

    tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
    logger.info("Converted Tokens: %s", tokens)
    # ['[CLS]', 'token_model', '##izing', 'text', 'is', 'a', 'core', 'task', 'of', 'nl', '##p', '[SEP]']


def get_tokenizer(model_id, truncation_side: str=None, chat_template=None, pad_token_id=None, max_length=None) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    if max_length is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=max_length)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None:
        if pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = pad_token_id

    if truncation_side is not None:
        tokenizer.truncation_side = truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if chat_template is not None:
        tokenizer.chat_template = chat_template
    elif hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def prepare_optimizer_and_scheduler(model, learning_rate, num_warmup_steps, num_training_steps):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [
                p
                for n, p in list(model.named_parameters())
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in list(model.named_parameters())
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(params=param_groups, lr=learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def display_model_info(model):
    # Number of parameters
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of Parameters: {num_parameters / 1000000:.2f} M")

    # Number of transformer layers
    num_layers = model.config.num_hidden_layers
    print(f"Number of Transformer Layers: {num_layers}")

    # Hidden dimension size
    hidden_size = model.config.hidden_size
    print(f"Hidden Dimension Size: {hidden_size}")

    # Attention head size
    num_attention_heads = model.config.num_attention_heads
    print(f"Number of Attention Heads: {num_attention_heads}")

    # Intermediate (feed-forward) hidden dimension size
    if hasattr(model.config, "intermediate_size"):
        intermediate_size = model.config.intermediate_size
        print(f"Intermediate (Feed-forward) Hidden Dimension Size: {intermediate_size}")

    # Type of model architecture (e.g., 'bert', 'gpt2', 'roberta', etc.)
    model_type = model.config.model_type
    print(f"Model Type: {model_type}")

    # Whether the model uses absolute position embeddings
    if hasattr(model.config, "position_embedding_type"):
        position_embedding_type = model.config.position_embedding_type
        print(f"Position Embedding Type: {position_embedding_type}")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def get_lora_config(config):
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,  # a higher alpha value assigns more weight to the LoRA activations
        target_modules=config.lora.target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=config.lora.task_type
    )
    return lora_config


def merge_adapter_and_save(adapter_dir, output_dir):
    config = PeftConfig.from_pretrained(adapter_dir)

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_dir)  # model param: base model

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)


def save_model_tokenizer(model, tokenizer, output_dir):
    print("=> Saving model and tokenizer to {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint = {"state_dict": model.state_dict()}
    torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pth.tar"))
    model.config.save_pretrained(output_dir)  # config is specific to Huggingface transformers
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    merge_adapter_and_save("/home/chanys/tnlp/expts/ultrachat_sft", "/home/chanys/tnlp/expts/ultrachat_merged_model")