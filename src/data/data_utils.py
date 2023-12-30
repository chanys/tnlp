from typing import List, Literal
import json
import re

from datasets import ClassLabel, Dataset, Features, Value, Sequence


def apply_chat_template(
    example, tokenizer, task: Literal["sft", "generation", "rm", "dpo"] = "sft", assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            # Insert system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
            # TODO: handle case where chosen/rejected also have system messages
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
            example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def read_sequence_examples_from_file(filepath: str):
    """Read sequence_model classification examples from a JSONL file to Dataset
    REF https://huggingface.co/docs/datasets/v1.1.1/loading_datasets.html#from-a-python-dictionary
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    texts = [d["text"] for d in data]

    has_label_field = all(
        "label" in d for d in data
    )  # does every example have a label field?

    if has_label_field:
        label_strings = [d["label"] for d in data]
        label_types = sorted(list(set(label_strings)))
        class_label = ClassLabel(num_classes=len(label_types), names=label_types)

        data_dict = {
            "text": texts,
            "label": label_strings,
        }

        features = Features(
            {
                "text": Value(dtype="string"),
                "label": ClassLabel(num_classes=len(label_types), names=label_types),
            }
        )

    else:
        data_dict = {"text": texts}
        features = Features({"text": Value(dtype="string")})

    ds = Dataset.from_dict(mapping=data_dict, features=features)
    return ds


def read_token_examples_from_file(filepath: str) -> Dataset:
    """Read token_model classification examples from a JSONL file to Dataset
    REF https://huggingface.co/docs/datasets/about_dataset_features
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    tokens = [d["tokens"] for d in data]  # type: List[List[str]]

    has_label_field = all(
        "labels" in d for d in data
    )

    if has_label_field:
        label_strings = [d["labels"] for d in data]  # type: List[List[str]]

        label_types = sorted(
            list(set(token_label for labels in label_strings for token_label in labels))
        )
        #index2label = {index: label for index, label in enumerate(label_types)}
        #label2index = {label: index for index, label in enumerate(label_types)}

        #label_indices = [
        #    [label2index[token_label] for token_label in labels] for labels in label_strings
        #]

        #class_label = ClassLabel(num_classes=len(label_types), names=label_types)

        data_dict = {
            "tokens": tokens,
            #"labels": label_indices,
            "labels": label_strings,
        }
        features = Features(
            {
                "tokens": Sequence(feature=Value(dtype="string")),
                #"labels": Sequence(feature=Value(dtype="int64")),
                #"label_strings": Sequence(feature=Value(dtype="string")),
                #"class_label": class_label,
                "labels": Sequence(feature=ClassLabel(num_classes=len(label_types), names=label_types)),
            }
        )
    else:
        data_dict = {"tokens": tokens}
        features = Features({"tokens": Sequence(feature=Value(dtype="string"))})

    ds = Dataset.from_dict(mapping=data_dict, features=features)
    return ds


def read_seq2seq_examples_from_file(filepath: str):
    """Read seq2seq examples from a JSONL file to Dataset
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    contexts = [d["context"] for d in data]

    has_response_field = all("response" in d for d in data)

    if has_response_field:
        responses = [d["response"] for d in data]

        data_dict = {
            "context": contexts,
            "response": responses,
        }

        features = Features(
            {
                "context": Value(dtype="string"),
                "response": Value(dtype="string"),
            }
        )
    else:
        data_dict = {"context": contexts}
        features = Features({"context": Value(dtype="string")})

    ds = Dataset.from_dict(mapping=data_dict, features=features)
    return ds


def read_chat_examples_from_file(filepath: str):
    """Read chat examples from a JSONL file to Dataset
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    # each d['text'] is a List[Dict], where each Dict is: {'content': "...", 'role': <role>}
    # <role> is either 'user' or 'assistant'
    messages = [d["text"] for d in data]
    ds = Dataset.from_dict({"messages": messages})
    return ds


def read_preference_examples_from_file(filepath: str):
    """Read preference examples from a JSONL file to Dataset
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    # each d['chosen'] is a List[Dict], where each Dict is: {'content': "...", 'role': <role>}
    # <role> is either 'user' or 'assistant'
    # similar for d['rejected']
    chosen_texts = [d["chosen"] for d in data]
    rejected_texts = [d["rejected"] for d in data]
    ds = Dataset.from_dict({"chosen": chosen_texts, "rejected": rejected_texts})
    return ds


def read_nyth_relation_examples_from_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    label = [d["label"] for d in data]
    tokens = [d["tokens"] for d in data]
    head_token_start = [d["head"]["token_start"] for d in data]
    head_token_end = [d["head"]["token_end"] for d in data]
    tail_token_start = [d["tail"]["token_start"] for d in data]
    tail_token_end = [d["tail"]["token_end"] for d in data]

    label_types = sorted(list(set(label)))

    features = Features(
            {
                "tokens": Sequence(feature=Value(dtype="string")),
                "label": ClassLabel(num_classes=len(label_types), names=label_types),
                "head_token_start": Value(dtype="int64"),
                "head_token_end": Value(dtype="int64"),
                "tail_token_start": Value(dtype="int64"),
                "tail_token_end": Value(dtype="int64"),
            }
        )

    ds = Dataset.from_dict(
        {"label": label, "tokens": tokens, "head_token_start": head_token_start, "head_token_end": head_token_end,
         "tail_token_start": tail_token_start, "tail_token_end": tail_token_end}, features=features)
    return ds


def read_bioasq_examples_from_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f.readlines()]

    query_strings = []
    chosen_strings = []
    rejected_strings = []
    for data in datas:
        query = data["query"]
        chosen = data["chosen"]
        for rejected in data["rejected"]:
            query_strings.append(query)
            chosen_strings.append(chosen)
            rejected_strings.append(rejected)

    mapping = {"query": query_strings, "chosen": chosen_strings, "rejected": rejected_strings}
    ds = Dataset.from_dict(mapping=mapping)
    return ds


def read_bioasq_as_triplet_examples_from_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f.readlines()]

    anchor_strings = []
    chosen_strings = []
    rejected_strings = []
    anchor_starts = []
    anchor_ends = []
    for data in datas:
        anchor = data["query"]
        chosen = data["chosen"]
        for rejected in data["rejected"]:
            anchor_strings.append(anchor)
            chosen_strings.append(chosen)
            rejected_strings.append(rejected)
            # TODO: currently using anchor_start=0 and anchor_end=0 , which will go on to select the [CLS] token.
            # TODO: should replace with real character offsets when we have a suitable dataset for entity linking
            anchor_starts.append(0)
            anchor_ends.append(0)

    mapping = {"anchor": anchor_strings, "chosen": chosen_strings, "rejected": rejected_strings, "anchor_start": anchor_starts, "anchor_end": anchor_ends}
    ds = Dataset.from_dict(mapping=mapping)
    return ds


def read_rag_examples_from_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f.readlines()]

    query = [data["query"] for data in datas]
    passage = [data["passage"] for data in datas]
    answer = [data["answer"] for data in datas]

    mapping = {"query": query, "passage": passage, "answer": answer}
    ds = Dataset.from_dict(mapping=mapping)
    return ds


