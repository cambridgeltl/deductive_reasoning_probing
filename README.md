# Can Pretrained Language Models (Yet) Reason Deductively?
**Authors**: Zhangdie Yuan, Songbo Hu, Ivan Vulić, Anna Korhonen and Zaiqiao Meng

This repository is constantly being updated. Contributions, feedbacks, comments and suggestions are welcome!

If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!

```latex
@inproceedings{yuan2023can,
  title={Can Pretrained Language Models (Yet) Reason Deductively?},
  author={Yuan, Zhangdie and Hu, Songbo and Vuli{\'c}, Ivan and Korhonen, Anna and Meng, Zaiqiao},
  booktitle={The 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2023)},
  year={2023}
}
```



## News

- **[January 22, 2023]**  This work has been accepted to appear at the conference (main) of EACL 2023.
- **[October 12, 2022]**  The preprint of this work is available at [Arxiv](https://arxiv.org/abs/2210.06442).

## Installation
This repository is tested on Python 3.8+, PyTorch 1.3.1+, transformers 4.20.0+, sentence-transformer 2.2.0+, cuda 11.3+, and nVIDIA RTX A6000 GPUs. We recommend using a virtual environment to set up the required packages and their dependencies via

```
$(env) pip install -r requirements.txt
```
## Quick Start
A number of scripts are provided for reproducibility with ease. For example, to reproduce the results of the classification task, one simply needs to firstly train the model via
```
$(env) bash ./scripts/lot_train_classification.sh
```
and then inference via
```
$(env) ./scripts/lot_test_classification.sh
```

which call `src/train_classification.py` and `src/test_classification.py` correspondingly, where hyperparameters are presented and can be adjusted accordingly. Some additional details such as data construction, data pre-processing and supplementary probing experiments can be found in `src/preprocessing.py` and `src/probe.py`.


## Data
The `data` folder contains all data, namely `LeapOfThought` and `WikiData`, that are required to reproduce our models in the paper. All data are after pre-processing as the original raw data might not be presented in an anonymous manner. We do not claim the ownership of `LeapOfThought` where credits to [alontalmor/LeapOfThought](https://github.com/alontalmor/LeapOfThought).

## Hyperparameters
All hyperparameters are set to the default values unless otherwise specified in the script or its corresponding python source code. For example, the following script in `scripts/lot_train_mlm.sh`
```
python src/train_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/LeapOfThought/fine-tuning/lot_train_mlm.json \
    --validation_file data/LeapOfThought/fine-tuning/lot_dev_mlm.json \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 2 \
    --line_by_line \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir training_output/lot-bert-base-uncased
```
passes hyperparameters via the `HfArgumentParser` from [transformers](https://github.com/huggingface/transformers).


For classification-based reasoning, we tested the following models with their hypermeters:

| pre-trained checkpoint  | batch | training epoch |
|-------------------------|-------|----------------|
| distilbert-base-uncased | 64    | 20             |
| bert-base-uncased       | 64    | 20             |
| bert-large-uncased      | 32    | 20             |
| roberta-base            | 64    | 20             |
| roberta-large           | 32    | 20             |

We choose the checkpoint with the best validation performance.



### Other resources

For more natural language reasoning papers, please refer to this repo: [Awesome Natural Language Reasoning Papers](https://github.com/mengzaiqiao/awesome-natural-language-reasoning)
