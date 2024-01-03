import os
import logging

import torch
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,
    set_seed, default_data_collator
)
from peft import get_peft_model, prepare_model_for_int8_training, PeftModel, PeftConfig

from src.model.seq2seq_model.seq2seq_utils import read_token_examples_to_seq2seq, parse_output_to_entities, \
    align_sequences, EntityOutput
from src.data.data_utils import read_seq2seq_examples_from_file
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
        if self.config.task == "seq2seq_ner":
            self.clean_up_tokenization_spaces = False
        else:
            self.clean_up_tokenization_spaces = True

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

    def read_examples(self, json_file):
        if self.config.task == "seq2seq_ner":
            return read_token_examples_to_seq2seq(json_file)  # this reads CoNLL BIO examples to seq2seq format
        else:
            return read_seq2seq_examples_from_file(json_file)

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

        ds = self.read_examples(self.config.data.inference_jsonl)
        ds_tokenized = ds.map(tokenize_context_response, fn_kwargs=fn_kwargs, batched=True,
                              batch_size=self.config.hyperparams.batch_size)

        dataloader = DataLoader(
            ds_tokenized,
            batch_size=self.config.hyperparams.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=False):
                outputs = model.generate(input_ids=torch.tensor(batch["input_ids"]).to(self.device),
                                         do_sample=True,
                                         top_p=0.9, max_new_tokens=self.config.hyperparams.max_response_length)

                for prediction in outputs:
                    prediction = tokenizer.decode(prediction.detach().cpu().numpy(), skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=self.clean_up_tokenization_spaces)
                    predictions.append(prediction)

        return ds["context"], predictions

    def _compute_score(self, predictions, references):
        if self.config.task == "seq2seq_ner":  # augmented output

            print(f"len(predictions)={len(predictions)}")
            print(f"len(references)={len(references)}")

            reference_count = 0
            predicted_count = 0
            common_count = 0
            for prediction, reference in zip(predictions, references):
                predicted_entities, predicted_tokens = parse_output_to_entities(prediction)

                reference_entities, reference_tokens = parse_output_to_entities(reference)

                print(f"prediction={prediction}, len(predicted_entities)={len(predicted_entities)}")
                for e in predicted_entities:
                    print(e)

                reference_set = set()
                for entity in reference_entities:
                    reference_set.add((entity.start, entity.end, " ".join(entity.labels)))

                token_matches = align_sequences(reference_tokens, predicted_tokens)

                predicted_set = set()
                print("len(predicted_set)=", len(predicted_set))
                for entity in predicted_entities:
                    new_start = None
                    new_end = None

                    for j in range(entity.start, entity.end + 1):
                        if j in token_matches:
                            if new_start is None:
                                new_start = token_matches[j]
                            new_end = token_matches[j]

                    if new_start is not None and new_end is not None:
                        predicted_set.add((new_start, new_end, " ".join(entity.labels)))

                common_count += len(reference_set.intersection(predicted_set))
                reference_count += len(reference_set)
                predicted_count += len(predicted_set)
                print(f"predicted_count={predicted_count}")

            precision = float(common_count) / predicted_count if predicted_count != 0 else 0.0
            recall = float(common_count) / reference_count if reference_count != 0 else 0.0
            if precision > 0 and recall > 0:
                f1 = (2 * precision * recall) / (precision + recall)
            else:
                f1 = 0

            print(f"common_count={common_count} predicted_count={predicted_count} reference_count={reference_count}")
            print(f"F1/P/R={f1:.3f}/{precision:.3f}/{recall:.3f}")

        else:  # raw text output
            rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)
            print(f"Rogue1: {rogue['rouge1'] * 100:2f}%")
            print(f"rouge2: {rogue['rouge2'] * 100:2f}%")
            print(f"rougeL: {rogue['rougeL'] * 100:2f}%")
            print(f"rougeLsum: {rogue['rougeLsum'] * 100:2f}%")

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

        ds_test = self.read_examples(self.config.data.test_jsonl)
        ds_test_tokenized = ds_test.map(tokenize_context_response, fn_kwargs=fn_kwargs, batched=True,
                                        batch_size=self.config.hyperparams.batch_size)

        test_dataloader = DataLoader(
            ds_test_tokenized,
            batch_size=self.config.hyperparams.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        predictions, references = [], []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, disable=False):
                outputs = model.generate(input_ids=torch.tensor(batch["input_ids"]).to(self.device),
                                         do_sample=True,
                                         top_p=0.9, max_new_tokens=self.config.hyperparams.max_response_length)

                for j, (input_ids, label_ids, prediction) in enumerate(
                        zip(batch['input_ids'], batch['labels'], outputs)):
                    prediction = tokenizer.decode(prediction.detach().cpu().numpy(), skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=self.clean_up_tokenization_spaces)
                    # np.where(condition, x, y): if condition==True, do x, else do y
                    labels = np.where(torch.tensor(label_ids) != -100, label_ids, tokenizer.pad_token_id)
                    labels = tokenizer.decode(labels, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=self.clean_up_tokenization_spaces)

                    predictions.append(prediction)
                    references.append(labels)

            # following is when doing this one by one
            # for sample in tqdm(ds_test_tokenized):
            #     # generate summary. unsqueeze(0): add an extra dimension to the tensor
            #     outputs = model.generate(input_ids=torch.tensor(sample["input_ids"]).unsqueeze(0).cuda(), do_sample=True,
            #                              top_p=0.9, max_new_tokens=self.config.hyperparams.max_response_length)
            #     prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=self.clean_up_tokenization_spaces)
            #
            #     # np.where(condition, x, y): if condition==True, do x, else do y
            #     labels = np.where(torch.tensor(sample['labels']) != -100, sample['labels'], tokenizer.pad_token_id)
            #     labels = tokenizer.decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=self.clean_up_tokenization_spaces)
            #
            #     predictions.append(prediction)
            #     references.append(labels)

        self._compute_score(predictions, references)

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer_id)

        fn_kwargs = {"tokenizer": tokenizer, "max_context_length": self.config.hyperparams.max_context_length,
                     "max_response_length": self.config.hyperparams.max_response_length,
                     "context_prompt": self.context_prompt, "response_prompt": self.response_prompt}

        ds_train = self.read_examples(self.config.data.train_jsonl)
        ds_train_tokenized = ds_train.map(tokenize_context_response, fn_kwargs=fn_kwargs, batched=True,
                                          batch_size=self.config.hyperparams.batch_size)

        ds_validation = self.read_examples(self.config.data.validation_jsonl)
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
            #compute_metrics=lambda pred: self.compute_metrics_for_eval(pred)
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
