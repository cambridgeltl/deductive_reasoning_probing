import random
import shutil
from itertools import chain

import sentence_transformers
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, models, evaluation, util, \
    LoggingHandler
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from torch import nn
import torch

from random import randrange, shuffle, sample

from sentence_transformers.cross_encoder import CrossEncoder

import configparser, codecs, math

import sys, os, time, csv, json
import numpy as np
import logging
from datasets import load_dataset

# Set random seed.
from util import ReasoningClassificationEvaluator

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)




def process_data(path):

    data = []
    f = open(path)
    for line in f:
        data.append(json.loads(line))
    f.close()

    data_samples = []

    for entry in data:
        text = entry["text"]
        label = 1 if entry["label"] == "True" else 0
        premise = text.split(" [SEP] ")[0]

        conclusion = text.split(" [SEP] ")[1]
        data_samples.append(InputExample(texts=[premise, conclusion], label=label))

    return data_samples


'''
Main function here
'''


def run_experiment():

    batch_size = 64
    epochs = 20
    init_model_path = "distilbert-base-uncased"
    note = "lot_train"

    save_model_path = init_model_path.replace("/", "-")
    train_samples = process_data(path="./data/LeapOfThought/classification/lot_train.json")
    assert len(train_samples) > 0

    model = CrossEncoder(init_model_path, num_labels=1)
    model.max_seq_length = 128

    # special_tokens = ["[SEP]"]
    # model.tokenizer.add_tokens(special_tokens, special_tokens=True)
    # model.model.resize_token_embeddings(len(model.tokenizer))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)


    evaluator_dev = ReasoningClassificationEvaluator(data_path="./data/LeapOfThought/classification/lot_dev.json")
    evaluator_test = ReasoningClassificationEvaluator(data_path="./data/LeapOfThought/classification/lot_test.json")
    evaluator = evaluation.SequentialEvaluator([evaluator_test, evaluator_dev], main_score_function=lambda scores: scores[-1])

    warmup_steps = math.ceil(len(train_dataloader) * epochs / batch_size * 0.1)

    output_path = "./output/lot-"+ save_model_path + "-e"+str(epochs)+"-bs"+ str(batch_size) +"-" + note+ "/"

    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=epochs,
              warmup_steps=warmup_steps,
              output_path=output_path)
    shutil.copyfile(sys.argv[0], output_path+"/code.py")


# Main program starts here
def main():
    run_experiment()

## MAIN starts here
if __name__ == '__main__':
    main()
