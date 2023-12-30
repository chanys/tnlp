import os
import json
import argparse
from typing import Dict
import re
import random

from datasets import load_dataset

from src.scripts.common_utils import Span, find_phrase_offsets, get_span_using_char_start, get_span_using_char_end


def convert_emotion_dataset_to_jsonl(output_dir):
    dataset_name = "emotion"
    ds = load_dataset(dataset_name)

    label_mapping = ds["train"].features["label"]

    splits = ["train", "validation", "test"]
    for split in splits:
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as o:
            for example in ds[split]:
                label_string = label_mapping.int2str(example["label"])
                example["label"] = label_string
                json.dump(example, o)
                o.write("\n")


def convert_conll2003_dataset_to_jsonl(output_dir):
    dataset_name = "conll2003"
    ds = load_dataset(dataset_name)

    label_mapping = ds["train"].features["ner_tags"].feature

    splits = ["train", "validation", "test"]
    for split in splits:
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as o:
            for example in ds[split]:
                label_strings = [label_mapping.int2str(tag) for tag in example["ner_tags"]]
                d = {"tokens": example["tokens"], "labels": label_strings}
                json.dump(d, o)
                o.write("\n")


def convert_samsum_dataset_to_jsonl(output_dir):
    dataset_name = "samsum"
    ds = load_dataset(dataset_name)

    splits = ["train", "validation", "test"]
    for split in splits:
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as o:
            for example in ds[split]:
                d = {"context": example["dialogue"], "response": example["summary"]}
                json.dump(d, o)
                o.write("\n")


def convert_ultrachat_dataset_to_jsonl(output_dir, splits: Dict[str, int]):
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    ds = load_dataset(dataset_name)

    for split, max_size in splits.items():
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as o:
            for index, example in enumerate(ds[split]):
                if index >= max_size:
                    break
                d = {"text": example["messages"]}
                json.dump(d, o)
                o.write("\n")


def convert_ultrafeedback_dataset_to_jsonl(output_dir, splits: Dict[str, int]):
    dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
    ds = load_dataset(dataset_name)

    for split, max_size in splits.items():
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as o:
            for index, example in enumerate(ds[split]):
                if index >= max_size:
                    break
                d = {"chosen": example["chosen"], "rejected": example["rejected"]}
                json.dump(d, o)
                o.write("\n")


def convert_nyth_dataset_to_json(input_basedir, output_dir):
    files = [os.path.join(input_basedir, "train_nonna.json"), os.path.join(input_basedir, "dev.json")]

    for input_file in files:
        filename = os.path.basename(input_file) + "l"

        outlines = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data = json.loads(line)
                label = data["relation"]

                text = re.sub(r"\s+", " ", data["sentence"])  # data['sentence'] seems to be already tokenized

                char_offset = 0
                token_spans = []
                for index, token in enumerate(text.split()):
                    token_spans.append(Span(token, index, index+1, char_offset, char_offset+len(token)))
                    char_offset += len(token) + 1  # +1 for the space

                head_offsets = find_phrase_offsets(data["head"]["word"], text)
                tail_offsets = find_phrase_offsets(data["tail"]["word"], text)

                if len(head_offsets) == 1 and len(tail_offsets) == 1:
                    head_offset = head_offsets[0]
                    tail_offset = tail_offsets[0]

                    head_span_start = get_span_using_char_start(token_spans, head_offset[0])
                    head_span_end = get_span_using_char_end(token_spans, head_offset[1])

                    tail_span_start = get_span_using_char_start(token_spans, tail_offset[0])
                    tail_span_end = get_span_using_char_end(token_spans, tail_offset[1])

                    if all(val is not None for val in [head_span_start, head_span_end, tail_span_start, tail_span_end]):

                        head_d = {"token_start": head_span_start.token_start, "token_end": head_span_end.token_end,
                                 "char_start": head_span_start.char_start, "char_end": head_span_end.char_end}
                        head_d["text"] = ' '.join(
                            span.text for span in token_spans[head_d["token_start"]: head_d["token_end"]])
                        tail_d = {"token_start": tail_span_start.token_start, "token_end": tail_span_end.token_end,
                                  "char_start": tail_span_start.char_start, "char_end": tail_span_end.char_end}
                        tail_d["text"] = ' '.join(
                            span.text for span in token_spans[tail_d["token_start"]: tail_d["token_end"]])

                        d = {"label": label, "text": text, "tokens": [token.text for token in token_spans], "head": head_d, "tail": tail_d}
                        outlines.append(d)

        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as o:
            for line in outlines:
                json.dump(line, o)
                o.write("\n")


def convert_bioasq_dataset_to_json(output_dir):
    with open("/home/chanys/data/BioASQ-training11b/training11b.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    outlines = []
    for example in data["questions"]:
        query = example["body"]
        chosen = example["ideal_answer"][0]
        if "exact_answer" in example and isinstance(example["exact_answer"], str) and example["exact_answer"] == "yes":
            rejected = [snippet["text"] for snippet in example["snippets"]]
            d = {"query": query, "chosen": chosen, "rejected": rejected}
            outlines.append(d)

    num_train = int(len(outlines) * 0.8)

    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as o:
        for line in outlines[:num_train]:
            json.dump(line, o)
            o.write("\n")

    with open(os.path.join(output_dir, "validation.jsonl"), "w", encoding="utf-8") as o:
        for line in outlines[num_train:]:
            json.dump(line, o)
            o.write("\n")


def convert_pubmed_qa_dataset_to_json(output_dir):
    ds = load_dataset('pubmed_qa', 'pqa_labeled')

    outlines = []
    for example in ds["train"]:
        query = example["question"]
        answer = example["long_answer"]
        passage = " ".join(text for text in example["context"]["contexts"])
        d = {"query": query, "passage": passage, "answer": answer}
        outlines.append(d)

    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as o:
        for line in outlines:
            json.dump(line, o)
            o.write("\n")


def convert_squadv2_dataset_to_json(output_dir, splits: Dict[str, int]):
    ds = load_dataset("squad_v2")

    for split, max_size in splits.items():
        outlines = []
        for example in ds[split]:
            query = example["question"]
            passage = example["context"]

            if len(example["answers"]["text"]) == 0:
                continue
            answer = example["answers"]["text"][0]

            if len(query.split()) > 16:
                continue
            if len(passage.split()) > 32:
                continue
            if len(query.split()) > 16:
                continue

            d = {"query": query, "passage": passage, "answer": answer}
            outlines.append(d)

        random.shuffle(outlines)
        outlines = outlines[:max_size]

        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding="utf-8") as o:
            for line in outlines:
                json.dump(line, o)
                o.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    random.seed(42)

    if args.dataset == "emotion":
        convert_emotion_dataset_to_jsonl(args.output_dir)
    elif args.dataset == "conll2003":
        convert_conll2003_dataset_to_jsonl(args.output_dir)
    elif args.dataset == "samsum":
        convert_samsum_dataset_to_jsonl(args.output_dir)
    elif args.dataset == "ultrachat":
        # limit size for experimental purposes, else the entire dataset is 2.5G
        splits = {"train_sft": 5000, "test_sft": 1000, "train_gen": 5000, "test_gen": 1000}
        convert_ultrachat_dataset_to_jsonl(args.output_dir, splits)
    elif args.dataset == "ultrafeedback":
        # limit size for experimental purposes
        splits = {"train_prefs": 500, "test_prefs": 500}
        convert_ultrafeedback_dataset_to_jsonl(args.output_dir, splits)
    elif args.dataset == "nyth":
        # relation extraction data from https://github.com/Spico197/NYT-H
        # https://aclanthology.org/2020.coling-main.566.pdf
        convert_nyth_dataset_to_json("/home/chanys/tnlp/raw_data/nyth", args.output_dir)
    elif args.dataset == "bioasq":
        convert_bioasq_dataset_to_json(args.output_dir)
    elif args.dataset == "pubmed_qa":
        convert_pubmed_qa_dataset_to_json(args.output_dir)
    elif args.dataset == "squad":
        splits = {"train": 500}
        convert_squadv2_dataset_to_json(args.output_dir, splits)
