import re
from pathlib import Path
from typing import List, Optional
import argparse

from pydantic import BaseModel, ConfigDict
import strictyaml
import yaml

import src.config

config_path = str(Path(src.config.__file__).resolve().parent)
base_dir = re.search(r"(.*)/src/config", config_path).group(1)


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    retriever_model_id: Optional[str] = None
    retriever_tokenizer_id: Optional[str] = None
    generator_model_id: Optional[str] = None
    generator_tokenizer_id: Optional[str] = None
    model_id: Optional[str] = None
    tokenizer_id: Optional[str] = None
    model_path: Optional[str] = None


class HyperParamsConfig(BaseModel):
    batch_size: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None

    gradient_accumulation_steps: Optional[int] = None
    epoch: Optional[int] = None
    learning_rate: Optional[float] = None
    warmup_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = None
    temperature: Optional[float] = None

    max_seq_length: Optional[int] = None
    max_context_length: Optional[int] = None  # for seq2seq
    max_response_length: Optional[int] = None  # for seq2seq
    max_query_len: Optional[int] = None  # for RAG
    max_passage_len: Optional[int] = None  # for RAG
    max_generator_len: Optional[int] = None  # for RAG
    max_prompt_length: Optional[int] = None

    encoder_dropout: Optional[int] = None
    linear_dropout: Optional[int] = None
    fcl_hidden_dim: Optional[int] = None


class LoraConfig(BaseModel):
    r: int
    alpha: int
    target_modules: List[str]
    task_type: str


class ProcessingConfig(BaseModel):
    retriever_output_dir: Optional[str] = None
    generator_output_dir: Optional[str] = None
    output_dir: Optional[str] = None
    merged_model_output_dir: Optional[str] = None
    seed: int


class DataConfig(BaseModel):
    train_jsonl: Optional[str] = None
    validation_jsonl: Optional[str] = None
    test_jsonl: Optional[str] = None
    inference_jsonl: Optional[str] = None


class PromptConfig(BaseModel):
    context_prompt: Optional[str] = None
    response_prompt: Optional[str] = None


class Config(BaseModel):
    """Master config object"""
    task: str
    model: ModelConfig
    hyperparams: HyperParamsConfig
    lora: Optional[LoraConfig] = None
    processing: Optional[ProcessingConfig] = None
    data: Optional[DataConfig] = None
    prompt: Optional[PromptConfig] = None


def replace_base_dir(value, base_dir: str):
    if isinstance(value, list):
        return [replace_base_dir(item, base_dir) for item in value]
    elif isinstance(value, dict):
        return {key: replace_base_dir(item, base_dir) for key, item in value.items()}
    elif isinstance(value, str):
        return value.replace("BASE_DIR", base_dir)
    else:
        return value

# def replace_base_dir(value, base_dir: str):
#     if isinstance(value, strictyaml.Seq):
#         return strictyaml.Seq([replace_base_dir(item, base_dir) for item in value])
#     elif isinstance(value, strictyaml.Map):
#         return strictyaml.Map({key: replace_base_dir(item, base_dir) for key, item in value.items()})
#     elif isinstance(value, strictyaml.Str):
#         replaced_value = value.data.replace("BASE_DIR", base_dir)
#         return strictyaml.Str(replaced_value)
#     else:
#         return value


def create_and_validate_config(config_filepath: str) -> Config:
    with open(config_filepath, "r", encoding="utf-8") as f:
        parsed_config = yaml.safe_load(f)

    parsed_config = replace_base_dir(parsed_config, base_dir)

    lora_config = LoraConfig(**parsed_config.get("lora")) if parsed_config.get("lora") else None
    processing_config = ProcessingConfig(**parsed_config.get("processing")) if parsed_config.get("processing") else None
    data_config = DataConfig(**parsed_config.get("data")) if parsed_config.get("data") else None
    prompt_config = PromptConfig(**parsed_config.get("prompt")) if parsed_config.get("prompt") else None

    _config = Config(
        task=parsed_config["task"],
        model=ModelConfig(**parsed_config["model"]),
        hyperparams=HyperParamsConfig(**parsed_config["hyperparams"]),
        lora=lora_config,
        processing=processing_config,
        data=data_config,
        prompt=prompt_config
    )

    return _config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = create_and_validate_config(args.config)
