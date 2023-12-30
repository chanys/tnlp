import logging
import os

import torch

from transformers import (
    TrainingArguments,
    set_seed, BitsAndBytesConfig, AutoModelForCausalLM
)
from peft import PeftModel, PeftConfig

from src.data.data_utils import apply_chat_template, read_preference_examples_from_file

from src.model.model_utils import get_lora_config, get_tokenizer

from trl import DPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_adapter_model(model_id: str) -> bool:
    if os.path.isdir(model_id):
        files = os.listdir(model_id)
        if "adapter_model.safetensors" in files or "adapter_model.bin" in files:
            return True
    return False


class DPOFineTuneLM(object):
    """DPO fine-tuning of a causal LM
    """

    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def inference(self):
        """Example for chat generation using the model trained via train() method
        """
        peft_config = PeftConfig.from_pretrained(self.config.model.model_id)

        tokenizer = get_tokenizer(self.config.model.model_id)

        model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=True,
                                                     device_map={"": 0})  # fit entire model on device 0

        model = PeftModel.from_pretrained(model, self.config.model.model_id, device_map={"": 0})
        model.eval()

        messages = [
            {"role": "user", "content": "Hello! I'm interested in buying a piano."},
            {"role": "assistant",
             "content": "That's great! Pianos come in various types and sizes. What kind of piano are you looking for?"},
            {"role": "user", "content": "I'm not sure. What are the different types available?"},
            {"role": "assistant",
             "content": "There are grand pianos, upright pianos, and digital pianos. Grand pianos have a more classic look, while digital pianos offer modern features. Upright pianos are a good compromise. Do you have a preference?"},
            {"role": "user", "content": "I think an upright piano would be suitable for my space."},
            {"role": "assistant",
             "content": "Excellent choice! Upright pianos are space-efficient and offer beautiful sound. Do you have a specific brand or budget in mind?"},
            {"role": "user",
             "content": "I'm looking for something of good quality but within a reasonable budget. Any recommendations?"}
        ]

        # add_generation_prompt tags on "<|assistant|>" at the end, prompting the LM to generate
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_tensors="pt")
        print(tokenizer.decode(tokenized_chat[0]))

        outputs = model.generate(input_ids=tokenized_chat, max_length=512)
        print(tokenizer.decode(outputs[0]))

    def test(self):
        pass

    def train(self):
        """Train a preference model via DPO"""
        # Truncate from left to ensure we don't lose labels in final turn
        tokenizer = get_tokenizer(self.config.model.tokenizer_id, truncation_side="left")

        fn_kwargs = {"tokenizer": tokenizer, "task": "dpo"}

        ds_train = read_preference_examples_from_file(self.config.data.train_jsonl)
        column_names = list(ds_train.features)
        ds_train_tokenized = ds_train.map(apply_chat_template, fn_kwargs=fn_kwargs, remove_columns=column_names)

        ds_train_tokenized = ds_train_tokenized.rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

        ds_validation = read_preference_examples_from_file(self.config.data.validation_jsonl)
        column_names = list(ds_validation.features)
        ds_validation_tokenized = ds_validation.map(apply_chat_template, fn_kwargs=fn_kwargs,
                                                    remove_columns=column_names)

        ds_validation_tokenized = ds_validation_tokenized.rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model_kwargs = dict(
            use_flash_attention_2=False,
            torch_dtype="auto",
            use_cache=False,
            device_map="auto",
            quantization_config=quantization_config,
        )

        model = self.config.model.model_id

        if is_adapter_model(model):
            print("**** loading the adapter model")
            # load the model, merge the adapter weights and unload the adapter
            # Note: to run QLora, you will need to merge the based model separately as the merged model in 16bit
            peft_config = PeftConfig.from_pretrained(model)

            model_kwargs = dict(
                use_flash_attention_2=False,
                torch_dtype="auto",
                use_cache=False,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                **model_kwargs,
            )
            model = PeftModel.from_pretrained(
                base_model, self.config.model.model_id
            )
            model.eval()
            model = model.merge_and_unload()
            model_kwargs = None

        training_args = TrainingArguments(
            output_dir=self.config.processing.output_dir,
            evaluation_strategy="epoch",
            learning_rate=self.config.hyperparams.learning_rate,
            # weight_decay=0.01,
            num_train_epochs=self.config.hyperparams.epoch,
            push_to_hub=False,
            per_device_train_batch_size=self.config.hyperparams.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.hyperparams.per_device_eval_batch_size,
            lr_scheduler_type="linear",
            optim="rmsprop",
            warmup_ratio=0.1,
            gradient_checkpointing=True,
            gradient_accumulation_steps=self.config.hyperparams.gradient_accumulation_steps,
            bf16=True
        )

        trainer = DPOTrainer(
            model,
            None,
            model_init_kwargs=model_kwargs,
            ref_model_init_kwargs=None,
            args=training_args,
            beta=0.1,
            train_dataset=ds_train_tokenized,
            eval_dataset=ds_validation_tokenized,
            tokenizer=tokenizer,
            max_length=self.config.hyperparams.max_seq_length,
            max_prompt_length=self.config.hyperparams.max_prompt_length,
            peft_config=get_lora_config(self.config),
        )

        # training
        logger.info("*** Train ***")
        train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # metrics on validation data
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        logger.info("*** Save model ***")
        trainer.save_model(self.config.processing.output_dir)
