import logging
from typing import Dict

import numpy as np

import evaluate
seqeval = evaluate.load("seqeval")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def confusion_matrix(labels, y_valid, y_preds):
    num_classes = len(labels)
    matrix = [[0] * num_classes for _ in range(num_classes)]

    for true_label, pred_label in zip(y_valid, y_preds):
        matrix[true_label][pred_label] += 1

    # Print the confusion matrix
    print("Confusion Matrix:")
    print("\t" + "\t".join(labels))
    for i in range(num_classes):
        print(labels[i] + "\t" + "\t".join(str(matrix[i][j]) for j in range(num_classes)))


def compute_seqeval_metric(p, index2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # since 2nd and subsequent subtokens (of each word) are tagged with -100. Thus only retrieve predictions pertaining for 1st subtoken of each word
    true_predictions = [
        [index2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [index2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    for true_prediction, true_label in zip(true_predictions, true_labels):
        assert len(true_prediction) == len(true_label)

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def compute_f1_metric(preds, labels, index2label: Dict[int, str], ignore_class=None):
    print(f"type(preds)={type(preds)}")
    print(f"type(labels)={type(labels)}")
    #labels = pred.label_ids
    #preds = pred.predictions.argmax(-1)

    # Filter out the class to ignore
    if ignore_class is not None:
        mask = labels != ignore_class
        labels = labels[mask]
        preds = preds[mask]

    # Calculate true positives, false positives, and false negatives for each class
    unique_classes = np.unique(np.concatenate((labels, preds)))
    f1_scores = {}

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for class_label in unique_classes:
        true_positives = np.sum((labels == class_label) & (preds == class_label))
        false_positives = np.sum((labels != class_label) & (preds == class_label))
        false_negatives = np.sum((labels == class_label) & (preds != class_label))

        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        precision = true_positives / (true_positives + false_positives + 1e-12)
        recall = true_positives / (true_positives + false_negatives + 1e-12)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        f1_scores[f"class_{index2label[class_label]}_f1"] = f1

    f1_scores["macro_average_f1"] = np.mean(list(f1_scores.values()))

    micro_precision = total_true_positives / (total_true_positives + total_false_positives + 1e-12)
    micro_recall = total_true_positives / (total_true_positives + total_false_negatives + 1e-12)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-12)
    f1_scores["micro_average_f1"] = micro_f1

    return f1_scores