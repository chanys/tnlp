import os
import logging

import torch
import numpy as np

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,
    set_seed
)
from peft import get_peft_model, prepare_model_for_int8_training, PeftModel, PeftConfig

from src.data.data_utils import read_token_examples_from_file, read_seq2seq_examples_from_file
from src.model.metric import compute_seqeval_metric
from src.model.model_utils import get_lora_config

import evaluate

metric = evaluate.load("rouge")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_context_response(sample, tokenizer, max_context_length: int, max_response_length: int, context_prompt: str,
                              response_prompt: str):
    contexts = [context_prompt.format(item) for item in sample["context"]]
    model_inputs = tokenizer(contexts, max_length=max_context_length, padding="max_length", truncation=True)

    if "response" in sample:
        responses = [response_prompt.format(item) for item in sample["response"]]
        labels = tokenizer(text_target=responses, max_length=max_response_length, padding="max_length", truncation=True)

        # -100 : we will ignore these subword tokens when calculating the loss
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]

    return model_inputs


class Seq2SeqLM(object):
    def __init__(self, configuration, context_prompt: str, response_prompt: str):
        self.config = configuration
        self.context_prompt = context_prompt
        self.response_prompt = response_prompt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.processing.seed) if hasattr(self.config.processing, "seed") else set_seed(42)

    def predict_batch(self, batch, tokenizer, model):
        inputs = tokenizer(
            batch["tokens"],
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
        )

        # let's keep track of the word_ids
        batch_word_ids = []
        for batch_index in range(len(batch["tokens"])):
            word_ids = np.array(inputs.word_ids(batch_index=batch_index))
            attention_mask = inputs.attention_mask[batch_index]
            batch_word_ids.append(word_ids[attention_mask == 1])

        # after this, type(inputs) will go from BatchEncoder to a Python Dict, and inputs.word_ids will not be available
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
            logits = output.logits
            pred_label_ids = torch.argmax(logits, dim=-1).cpu().numpy()

        batch_labels = []
        for batch_index in range(len(batch["tokens"])):
            word_ids = batch_word_ids[batch_index]
            subword_label_ids = pred_label_ids[batch_index]

            previous_word_id = None
            label_ids = []
            for index, word_id in enumerate(word_ids):
                if word_id is None or word_id == previous_word_id:  # only take the prediction from each first sub-word
                    pass
                else:
                    label_ids.append(subword_label_ids[index])
                previous_word_id = word_id
            batch_labels.append(label_ids)

        return {"tokens": batch["tokens"], "labels": batch_labels}

    def inference(self):
        peft_config = PeftConfig.from_pretrained(self.config.model.model_id)

        model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=True,
                                                      device_map={"": 0})  # fit entire model on device 0
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

        model = PeftModel.from_pretrained(model, self.config.model.model_id, device_map={"": 0})
        model.eval()

        fn_kwargs = {"tokenizer": tokenizer, "max_context_length": self.config.hyperparams.max_context_length,
                     "max_response_length": self.config.hyperparams.max_response_length,
                     "context_prompt": self.context_prompt, "response_prompt": self.response_prompt}

        ds = read_seq2seq_examples_from_file(self.config.data.inference_jsonl)
        ds_tokenized = ds.map(tokenize_context_response, fn_kwargs=fn_kwargs, batched=True,
                              batch_size=self.config.hyperparams.batch_size)

        predictions = []
        for sample in tqdm(ds_tokenized):
            outputs = model.generate(input_ids=torch.tensor(sample["input_ids"]).unsqueeze(0).cuda(), do_sample=True,
                                     top_p=0.9, max_new_tokens=self.config.hyperparams.max_response_length)
            prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
            predictions.append(prediction)

        return ds["context"], predictions

    def test(self):
        peft_config = PeftConfig.from_pretrained(self.config.model.model_id)

        model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=True,
                                                      device_map={"": 0})  # {"": 0}: fit entire model on device 0
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

        model = PeftModel.from_pretrained(model, self.config.model.model_id, device_map={"": 0})
        model.eval()

        fn_kwargs = {"tokenizer": tokenizer, "max_context_length": self.config.hyperparams.max_context_length,
                     "max_response_length": self.config.hyperparams.max_response_length,
                     "context_prompt": self.context_prompt, "response_prompt": self.response_prompt}

        ds_test = read_seq2seq_examples_from_file(self.config.data.test_jsonl)
        ds_test_tokenized = ds_test.map(tokenize_context_response, fn_kwargs=fn_kwargs, batched=True,
                                        batch_size=self.config.hyperparams.batch_size)

        predictions, references = [], []
        for sample in tqdm(ds_test_tokenized):
            # generate summary. unsqueeze(0): add an extra dimension to the tensor
            outputs = model.generate(input_ids=torch.tensor(sample["input_ids"]).unsqueeze(0).cuda(), do_sample=True,
                                     top_p=0.9, max_new_tokens=self.config.hyperparams.max_response_length)
            prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)

            # np.where(condition, x, y): if condition==True, do x, else do y
            labels = np.where(torch.tensor(sample['labels']) != -100, sample['labels'], tokenizer.pad_token_id)
            labels = tokenizer.decode(labels, skip_special_tokens=True)

            predictions.append(prediction)
            references.append(labels)

        # compute metric
        rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

        # print results
        print(f"Rogue1: {rogue['rouge1'] * 100:2f}%")
        print(f"rouge2: {rogue['rouge2'] * 100:2f}%")
        print(f"rougeL: {rogue['rougeL'] * 100:2f}%")
        print(f"rougeLsum: {rogue['rougeLsum'] * 100:2f}%")

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)

        fn_kwargs = {"tokenizer": tokenizer, "max_context_length": self.config.hyperparams.max_context_length,
                     "max_response_length": self.config.hyperparams.max_response_length,
                     "context_prompt": self.context_prompt, "response_prompt": self.response_prompt}

        ds_train = read_seq2seq_examples_from_file(self.config.data.train_jsonl)
        ds_train_tokenized = ds_train.map(tokenize_context_response, fn_kwargs=fn_kwargs, batched=True,
                                          batch_size=self.config.hyperparams.batch_size)

        ds_validation = read_seq2seq_examples_from_file(self.config.data.validation_jsonl)
        ds_validation_tokenized = ds_validation.map(tokenize_context_response, fn_kwargs=fn_kwargs, batched=True,
                                                    batch_size=self.config.hyperparams.batch_size)

        # from https://huggingface.co/docs/transformers/main_classes/quantization, which says:
        # - you cannot train 8-bit weights (not supported yet), but can use 8-bit models to train extra parameters
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model.model_id, load_in_8bit=True, device_map="auto")
        print(f"Loaded model in 8bit, memory footprint={model.get_memory_footprint()}")

        lora_config = get_lora_config(self.config)

        model = prepare_model_for_int8_training(model)  # prepare int-8 model for training

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,  # ignore tokenizer pad token in the loss
            pad_to_multiple_of=8
        )

        logging_steps = len(ds_train) // self.config.hyperparams.batch_size
        training_args = Seq2SeqTrainingArguments(
            disable_tqdm=False,
            evaluation_strategy="epoch",
            gradient_accumulation_steps=self.config.hyperparams.gradient_accumulation_steps,
            learning_rate=self.config.hyperparams.learning_rate,
            logging_steps=logging_steps,
            num_train_epochs=self.config.hyperparams.epoch,
            output_dir=self.config.processing.output_dir,
            per_device_eval_batch_size=self.config.hyperparams.batch_size,
            per_device_train_batch_size=self.config.hyperparams.batch_size,
            push_to_hub=False,
            save_strategy="steps",
            save_steps=self.config.hyperparams.save_steps,
            save_total_limit=self.config.hyperparams.save_total_limit,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=ds_train_tokenized,
            eval_dataset=ds_validation_tokenized,
        )

        # In model inference, e.g. auto-regressive inference/generation, once you generate the hidden states
        # for each time-step, you'll want to store/cache it, so that you won't need to regenerate them again for every
        # time step. But during training, you'll actually want all these hidden states to change during training.
        model.config.use_cache = False  # silence the warnings for training. Re-enable for inference!

        trainer.train()

        trainer.model.save_pretrained(  # this saves the LoRa adapter
            os.path.join(self.config.processing.output_dir, "final_model")
        )
        tokenizer.save_pretrained(
            os.path.join(self.config.processing.output_dir, "final_model")
        )
