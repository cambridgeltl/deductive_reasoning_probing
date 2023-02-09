import json
import logging
import os
import csv

from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

def process_eval_data(path):

    data = []
    f = open(path)
    for line in f:
        data.append(json.loads(line))
    f.close()

    all_premise = []
    all_conclusion = []
    all_label = []
    for entry in data:
        text = entry["text"]
        label = 1 if entry["label"] == "True" else 0

        if " [SEP] " not in text:
            premise = ""

            conclusion = text
            all_premise.append(premise)
            all_conclusion.append(conclusion)
            all_label.append(label)
        else:
            premise = text.split(" [SEP] ")[0]

            conclusion = text.split(" [SEP] ")[1]
            all_premise.append(premise)
            all_conclusion.append(conclusion)
            all_label.append(label)
    return all_premise,all_conclusion,all_label



class ReasoningClassificationEvaluator(SentenceEvaluator):


    def __init__(self, data_path : str, name: str = '', batch_size: int = 32, show_progress_bar: bool = False, write_csv: bool = True):

        st1, st2, labels = process_eval_data(data_path)

        self.sentences1 = st1
        self.sentences2 = st2
        self.labels = labels
        self.data_path = data_path

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        if "test" in data_path:
            self.pre_fix = "test"
        elif "dev" in data_path or "val" in data_path:
            self.pre_fix = "dev"
        else:
            self.pre_fix = ""


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Binary Accuracy Evaluation of the model on " + self.name + " dataset" + out_txt)

        acc = self.compute_metrices(model)
        logger.info("Acc on " + self.pre_fix + ": " +  str(acc))

        return acc


    def compute_metrices(self, model):
        sentences = list(map(lambda x :  x, zip(self.sentences1 ,self.sentences2)))
        preds = model.predict(sentences,show_progress_bar=False)

        labels = np.asarray(self.labels)

        preds_labels = list(map(lambda x : x >=  0.5, preds))

        correct_labels = preds_labels == labels
        acc = (np.sum(correct_labels)) / len(correct_labels)

        return acc




if __name__ == '__main__':
    this_evaluator = ReasoningClassificationEvaluator(data_path="../data/LeapOfThought/classification/lot_train.json")

    init_model_path = "bert-base-uncased"
    model = CrossEncoder(init_model_path, num_labels=1)
    this_evaluator.__call__(model)