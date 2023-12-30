import logging
from typing import Union

import torch
from datasets import load_dataset

from transformers import (
    TrainingArguments,
    set_seed, AutoModelForCausalLM, DataCollatorForSeq2Seq,
    Trainer, GenerationConfig
)
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig

from src.data.data_utils import read_chat_examples_from_file, apply_chat_template

from src.model.model_utils import get_lora_config, get_tokenizer, merge_adapter_and_save, print_trainable_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prompt_input = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)

prompt_no_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def tokenize(tokenizer, prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < 256
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_prompt(instruction: str, input: Union[None, str] = None, label: Union[None, str] = None):
    if input is None or input == "":
        prompt = prompt_no_input.format(instruction=instruction)
    else:
        prompt = prompt_input.format(instruction=instruction, input=input)

    if label is not None:
        prompt = f"{prompt}{label}"

    return prompt


def generate_and_tokenize_prompt(data_point, tokenizer, instruction_field, input_field, label_field):
    """instruction_field, input_field, label_field : dataset dependent
    E.g. for alpaca, these correspond to: "instruction", "input", "output"
    """
    full_prompt = generate_prompt(instruction=data_point[instruction_field], input=data_point[input_field],
                                  label=data_point[label_field])
    user_prompt = generate_prompt(instruction=data_point[instruction_field], input=data_point[input_field])

    full_prompt_tokenized = tokenize(tokenizer, full_prompt)
    user_prompt_tokenized = tokenize(tokenizer, user_prompt, add_eos_token=False)

    user_prompt_len = len(user_prompt_tokenized["input_ids"])

    full_prompt_tokenized["labels"] = [-100] * user_prompt_len + full_prompt_tokenized["labels"][user_prompt_len:]
    return full_prompt_tokenized


class InstructionFineTuneLM(object):
    """Instruction fine-tuning of a causal LM
    """

    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def inference(self):
        peft_config = PeftConfig.from_pretrained(self.config.model.model_id)
        tokenizer = get_tokenizer(peft_config.base_model_name_or_path)

        model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=True,
                                                     device_map="auto")

        model = PeftModel.from_pretrained(model, self.config.model.model_id)
        model.eval()

        # tokenizer.pad_token_id = model.config.pad_token_id

        instruction = "Tell me about deep learning and transformers."
        prompt = generate_prompt(instruction=instruction, input=None)

        inputs = tokenizer(prompt, return_tensors="pt")

        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(temperature=0.1, top_p=1.0, top_k=10, num_beams=3)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output)

    def test(self):
        pass

    def train(self):
        tokenizer = get_tokenizer(self.config.model.tokenizer_id, pad_token_id=0)

        ds = load_dataset("json", data_files=self.config.data.train_jsonl)

        token_fn_kwargs = {"tokenizer": tokenizer, "instruction_field": "instruction", "input_field": "input",
                           "label_field": "output"}
        ds_train_tokenized = ds["train"].shuffle().map(generate_and_tokenize_prompt, fn_kwargs=token_fn_kwargs)

        data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16
        # )

        model = AutoModelForCausalLM.from_pretrained(self.config.model.model_id, load_in_8bit=True, device_map="auto")
        # model = AutoModelForCausalLM.from_pretrained(params.model_id, quantization_config=bnb_config, device_map="auto")

        model = prepare_model_for_kbit_training(model)

        lora_config = get_lora_config(self.config)

        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)

        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.hyperparams.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.hyperparams.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.hyperparams.gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=self.config.hyperparams.epoch,
            learning_rate=self.config.hyperparams.learning_rate,
            output_dir=self.config.processing.output_dir,
            report_to=None
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=ds_train_tokenized,
        )

        trainer.train()
        trainer.model.save_pretrained(self.config.processing.output_dir)
        tokenizer.save_pretrained(self.config.processing.output_dir)
