from typing import List
import json

from datasets import Dataset, Features, Value
import numpy as np


def convert_bio_to_labeled_tuples(labels: List[str]):
    """ From a list of BIO tags, condense into tuples of (start-token-index, end-token-index, tag)
    E.g.:
    index:  0  1     2       3    4          5         6    7    8
    token: In an explosion   ,  many       people    will  get  hurt
    labels: O  O     O       O  B-Patient  I-Patient   O    O    O

    We generate: (4, 5, Patient)
    """
    matches = []
    i = 0

    while i < len(labels):
        if labels[i].startswith("B-"):
            start, end = i, i + 1

            while end < len(labels) and (labels[end] == "I-" + labels[start][2:]):
                end += 1

            matches.append((start, end - 1, labels[start][2:]))
            i = end
        else:
            i += 1

    return matches


class SpanNode:
    def __init__(self, start: int, end: int, label: str = None):
        self.start = start
        self.end = end
        self.label = label
        self.children = []
        self.parent = None

    def __str__(self):
        return f"({self.start}, {self.end}, {self.label}, {len(self.children)})"


def convert_token_example_to_seq2seq(tokens: List[str], labels: List[str]):
    assert len(tokens) == len(labels)

    labeled_tuples = convert_bio_to_labeled_tuples(labels)

    # sort by (ascending) start-position. If this is the same, then sort by (decending) end-position.
    # this ensures if span A encompasses span B, then span A will be in front of span B in the sorted list.
    labeled_tuples = sorted(labeled_tuples, key=lambda x: (x[0], -x[1]))

    # convert each tuple to a SpanNode
    span_nodes = [SpanNode(start=start, end=end, label=label) for (start, end, label) in labeled_tuples]

    # add parent-child relations
    for index, span_node in enumerate(span_nodes):
        for i in range(index - 1, -1, -1):
            parent_candidate = span_nodes[i]
            if parent_candidate.start <= span_node.start and span_node.end <= parent_candidate.end:
                parent_candidate.children.append(span_node)
                span_node.parent = parent_candidate
                break

    root_node = SpanNode(start=0, end=len(tokens) - 1)
    for span_node in span_nodes:
        if span_node.parent is None:
            root_node.children.append(span_node)

    return span_node_tree_to_augmented_output(root_node, tokens)


def span_node_tree_to_augmented_output(node: SpanNode, tokens: List[str]) -> str:
    if len(node.children) == 0:
        node_tokens_string = ' '.join(tokens[i] for i in range(node.start, node.end + 1))
        return f"[ {node_tokens_string} | {node.label} ]"

    start = node.start

    ret = []

    for child in node.children:
        child_output = span_node_tree_to_augmented_output(child, tokens)

        for i in range(start, child.start):
            ret.append(tokens[i])

        ret.append(child_output)
        start = child.end + 1

    for i in range(start, node.end + 1):
        ret.append(tokens[i])

    if node.label is not None:
        ret.append("|")
        ret.append(node.label)

    return ' '.join(ret)


def read_token_examples_to_seq2seq(filepath: str) -> Dataset:
    """Read token_model classification examples from a JSONL file to Dataset
    REF https://huggingface.co/docs/datasets/about_dataset_features
    """
    with open(filepath, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f.readlines()]

    has_label_field = all("labels" in d for d in examples)

    if has_label_field:
        contexts = []
        responses = []
        for example in examples:
            assert "tokens" in example
            assert "labels" in example

            output_format = convert_token_example_to_seq2seq(example["tokens"], example["labels"])

            contexts.append(" ".join(example["tokens"]))
            responses.append(output_format)

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
        contexts = [" ".join(example["tokens"]) for example in examples]
        data_dict = {"context": contexts}
        features = Features({"context": Value(dtype="string")})

    ds = Dataset.from_dict(mapping=data_dict, features=features)
    return ds


class EntityOutput:
    def __init__(self, start: int, end: int = None, texts: List[str] = None, labels: List[str] = None):
        self.start = start
        self.end = end
        self.texts = texts if texts is not None else []
        self.labels = labels if labels is not None else []
        self.seen_separator = False

    def __str__(self):
        return f"({self.start},{self.end}) text=\"{' '.join(self.texts)}\" label=\"{' '.join(self.labels)}\""


def parse_output_to_entities(output: str):
    ret = []

    output_tokens = []
    entity_stack: List[EntityOutput] = []

    values = output.split()
    for value in values:
        if value == "[":  # start entity
            entity_stack.append(EntityOutput(len(output_tokens)))
        elif value == "]":  # end entity
            if len(entity_stack) > 0:
                entity = entity_stack.pop()
                entity.end = len(output_tokens) - 1
                ret.append(entity)
        else:
            if value == "|" and len(entity_stack) > 0:
                entity_stack[-1].seen_separator = True
            else:  # normal token
                # we need to check whether this is a text token, or a label token
                if len(entity_stack) > 0:
                    if entity_stack[-1].seen_separator:  # this is a label token
                        entity_stack[-1].labels.append(value)
                        continue

                    for entity in reversed(entity_stack):  # this is a text token
                        entity.texts.append(value)

                output_tokens.append(value)

    return ret, output_tokens


def align_sequences(reference_tokens, predicted_tokens):
    # now we align self.tokens with output_tokens (with dynamic programming)
    cost = np.zeros((len(reference_tokens) + 1, len(predicted_tokens) + 1))  # cost of alignment between tokens[:i]
    # and output_tokens[:j]
    best = np.zeros_like(cost, dtype=int)  # best choice when aligning tokens[:i] and output_tokens[:j]

    for i in range(len(reference_tokens) + 1):
        for j in range(len(predicted_tokens) + 1):
            if i == 0 and j == 0:
                continue

            candidates = []

            # match
            if i > 0 and j > 0:
                candidates.append(
                    ((0 if reference_tokens[i - 1] == predicted_tokens[j - 1] else 1) + cost[i - 1, j - 1], 1))

            # skip in the first sequence
            if i > 0:
                candidates.append((1 + cost[i - 1, j], 2))

            # skip in the second sequence
            if j > 0:
                candidates.append((1 + cost[i, j - 1], 3))

            chosen_cost, chosen_option = min(candidates)
            cost[i, j] = chosen_cost
            best[i, j] = chosen_option

    # reconstruct best alignment
    matching = {}

    i = len(reference_tokens) - 1
    j = len(predicted_tokens) - 1

    while i >= 0 and j >= 0:
        chosen_option = best[i + 1, j + 1]

        if chosen_option == 1:
            # match
            matching[j] = i
            i, j = i - 1, j - 1

        elif chosen_option == 2:
            # skip in the first sequence
            i -= 1

        else:
            # skip in the second sequence
            j -= 1

    return matching


