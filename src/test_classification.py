import glob
import random

from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, models, evaluation, util, \
    LoggingHandler

import torch

from random import randrange, shuffle, sample

from sentence_transformers.cross_encoder import CrossEncoder

import configparser, codecs, math

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

def eval_model(model_path):

    all_evalutor = []

    for item in glob.glob('../data/LeapOfThought/classification/*test.json'):
        test_evaluator = ReasoningClassificationEvaluator(
            data_path=item,
            name=item)
        all_evalutor.append(test_evaluator)


    evaluator = evaluation.SequentialEvaluator(all_evalutor, main_score_function=lambda scores: scores[-1])

    model = CrossEncoder(model_path)
    evaluator.__call__(model)


if __name__ == '__main__':

    model_paths = []
    model_path = "path to the model check point"
    for model_path in model_paths:
        logger.info("Testing models on :" + model_path)

        eval_model(model_path)
