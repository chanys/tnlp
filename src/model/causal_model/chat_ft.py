import logging

import torch

from transformers import (
    TrainingArguments,
    set_seed, BitsAndBytesConfig, AutoModelForCausalLM
)
from peft import PeftModel, PeftConfig

from src.data.data_utils import read_chat_examples_from_file, apply_chat_template

from src.model.model_utils import get_lora_config, get_tokenizer, merge_adapter_and_save

from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatFineTuneLM(object):
    """Fine tune a causal LM for chat
    """
    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def inference(self):
        """Example for chat generation using the model trained via train() method
        """
        peft_config = PeftConfig.from_pretrained(self.config.model.model_id)

        tokenizer = get_tokenizer(peft_config.base_model_name_or_path)

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
        """Train a chat model"""
        tokenizer = get_tokenizer(self.config.model.tokenizer_id)

        fn_kwargs = {"tokenizer": tokenizer, "task": "sft"}

        ds_train = read_chat_examples_from_file(self.config.data.train_jsonl)
        ds_train_tokenized = ds_train.map(apply_chat_template, fn_kwargs=fn_kwargs)

        ds_validation = read_chat_examples_from_file(self.config.data.validation_jsonl)
        ds_validation_tokenized = ds_validation.map(apply_chat_template, fn_kwargs=fn_kwargs)

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

        training_args = TrainingArguments(
            output_dir=self.config.processing.output_dir,
            evaluation_strategy="epoch",
            learning_rate=self.config.hyperparams.learning_rate,
            #weight_decay=0.01,
            num_train_epochs=self.config.hyperparams.epoch,
            push_to_hub=False,
            per_device_train_batch_size=self.config.hyperparams.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.hyperparams.per_device_eval_batch_size,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            gradient_accumulation_steps=self.config.hyperparams.gradient_accumulation_steps,
            bf16=True
        )

        trainer = SFTTrainer(
            model=self.config.model.model_id,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=ds_train_tokenized,
            eval_dataset=ds_validation_tokenized,
            dataset_text_field="text",
            max_seq_length=self.config.hyperparams.max_seq_length,
            tokenizer=tokenizer,
            packing=True,
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
        trainer.save_model(self.config.processing.output_dir)  # save the adapter

        # merge adapter weights and save the merged full model
        merge_adapter_and_save(self.config.processing.output_dir, self.config.processing.merged_model_output_dir)





