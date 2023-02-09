import torch
import pandas as pd
import sys
from transformers import pipeline
from pprint import pprint
import json

""" GPU setup """
if torch.cuda.is_available():
    device = torch.device('cuda')

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

import warnings
warnings.filterwarnings("ignore")

def remove_periods():
    path = 'data/LeapOfThought/mask-filling'
    with open(f'{path}/lot_test.txt') as input, \
         open(f'{path}/lot_no_periods_test.txt', 'w+') as output:
        for input_line in input.readlines():
            line = input_line.replace('.', '')
            output.write(f'{line}')

    path = 'data/WikiData/mask-filling'
    with open(f'{path}/wd_test_50.txt') as input, \
         open(f'{path}/wd_no_periods_test_50.txt', 'w+') as output:
        for input_line in input.readlines():
            line = input_line.replace('.', '')
            output.write(f'{line}')

def random_order():
    import random
    random.seed(0)

    path = 'data/LeapOfThought/classification'
    with open(f'{path}/lot_test.json') as input, \
         open(f'{path}/lot_random_order_fix_test.json', 'w+') as output:
        for input_line in input.readlines():
            instance = json.loads(input_line)
            text = instance['text']
            label = instance['label']
            context, conclusion = text.split(' [SEP] ')
            line_segs = context.split('.')
            random.shuffle(line_segs)
            context = '.'.join(line_segs)
            text_to_write = f'{context} [SEP] {conclusion}'
            dict_to_write = {'text': text_to_write, 'label': label}
            output.write(json.dumps(dict_to_write))
            output.write('\n')

    path = 'data/LeapOfThought/mask-filling'
    with open(f'{path}/lot_test.txt') as input, \
         open(f'{path}/lot_random_order_fix_test.txt', 'w+') as output:
        for input_line in input.readlines():
            text, target = input_line.strip().split('\t')
            line_segs = text.split('.')
            masked = [seg for seg in line_segs if '<MASK>' in seg]
            line_segs = [seg for seg in line_segs if '<MASK>' not in seg]   
            random.shuffle(line_segs)
            line_segs = line_segs + masked
            line = '.'.join(line_segs)
            output.write(f'{line}\t{target}\n')

    path = 'data/LeapOfThought/mask-filling'
    with open(f'{path}/lot_test.txt') as input, \
         open(f'{path}/lot_random_order_test.txt', 'w+') as output:
        for input_line in input.readlines():
            text, target = input_line.strip().split('\t')
            line_segs = text.split('.')
            random.shuffle(line_segs)
            line = '.'.join(line_segs)
            output.write(f'{line}\t{target}\n')
        
    path = 'data/WikiData/mask-filling'
    with open(f'{path}/wd_test_50.txt') as input, \
         open(f'{path}/wd_random_order_fix_test_50.txt', 'w+') as output:
        for input_line in input.readlines():
            text, target = input_line.strip().split('\t')
            line_segs = text.split('.')
            masked = [seg for seg in line_segs if '<MASK>' in seg]
            line_segs = [seg for seg in line_segs if '<MASK>' not in seg]
            random.shuffle(line_segs)
            line_segs = line_segs + masked
            line = '.'.join(line_segs)
            output.write(f'{line}\t{target}\n')

    path = 'data/WikiData/mask-filling'
    with open(f'{path}/wd_test_50.txt') as input, \
         open(f'{path}/wd_random_order_test_50.txt', 'w+') as output:
        for input_line in input.readlines():
            text, target = input_line.strip().split('\t')
            line_segs = text.split('.')
            random.shuffle(line_segs)
            line = '.'.join(line_segs)
            output.write(f'{line}\t{target}\n')

def neutral():
    path = 'data/WikiData/mask-filling'
    with open(f'{path}/wd_neutral_sent.txt') as input, \
         open(f'{path}/wd_neutral_test.txt', 'w+') as output:
        for input_line in input.readlines()[1:]:
            line = input_line.lower().strip()
            source, sent, _ = line.split('\t')
            masked = sent.replace(source, '<MASK>')
            target = source
            if masked.count('<MASK>') == 1:
                output.write(f'{masked}\t{target}\n')

def neutral_classification():
    path = 'data/LeapOfThought/classification'
    with open(f'{path}/lot_neutral_sent_3k.txt') as input, \
         open(f'{path}/lot_neutral_test.json', 'w+') as output:
        for input_line in input.readlines()[1:]:
            line = input_line.lower().strip()
            source, sent, _ = line.split('\t')
            output.write(json.dumps({'text': sent, 'label': 'True'}))
            output.write(f'\n')

def wikidata_split(max_num):
    path = 'data/WikiData/mask-filling'

    from sklearn.model_selection import train_test_split
    data = pd.read_csv(f'{path}/wd_all.txt', sep='\t', names=['text', 'target'])
    targets = pd.unique(data['target'])
    train_targets, dev_targets = train_test_split(targets, test_size=0.2, random_state=0)
    dev_targets, test_targets = train_test_split(dev_targets, test_size=0.5, random_state=0)

    splits = {'train': train_targets, 'dev': dev_targets, 'test': test_targets}

    for file_name in ['wd_all.txt', 'wd_nla_para_all.txt', 'wd_nla_syn_all.txt', 'wd_nla_syn_det_all.txt', 'wd_pegasus_para_all.txt', 'wd_pred_negation_all.txt']:
        data = pd.read_csv(f'{path}/{file_name}', sep='\t', names=['text', 'target'])
        for split in splits:
            split_targets = []
            for target in splits[split]:
                split_target_data = data[data['target']==target][:max_num]
                for _, row in split_target_data.iterrows():
                    cur_text, cur_target = row['text'], row['target']
                    split_targets.append([cur_text, cur_target])
            output_file_name = file_name.replace('all', split).replace('.txt', f'_{max_num}.txt')
            pd.DataFrame(split_targets).to_csv(path_or_buf=f'{path}/{output_file_name}', sep='\t', header=False, index=False)

def wikidata_split_all():
    for max_num in [50, 100, 500, 1000, 5000, 10000, 50000]:
        wikidata_split(max_num)

def wikidata_prepare_training():
    from os import listdir
    from os.path import isfile, join

    input_path = 'data/WikiData/mask-filling'
    output_path = 'data/WikiData/fine-tuning'
    all_input_files = [f for f in listdir(input_path) if isfile(join(input_path, f)) if '3k' not in f]

    for input_file_name in all_input_files:
        output_file_name = input_file_name.replace('.txt', '.json')
        with open(f'{input_path}/{input_file_name}') as input_file, \
             open(f'{output_path}/{output_file_name}', 'w+') as output_file:
             for line in input_file.readlines():
                 text, target = line.strip().split('\t')
                 output_line = text.replace('<MASK>', target)
                 output_file.write(json.dumps({'text': output_line}))
                 output_file.write('\n')

def wikidata_prepare_training_specific_mask():
    from os import listdir
    from os.path import isfile, join

    input_path = 'data/WikiData/mask-filling'
    output_path = 'data/WikiData/fine-tuning'
    all_input_files = [f for f in listdir(input_path) if isfile(join(input_path, f)) if '3k' not in f]

    for input_file_name in all_input_files:
        output_file_name = input_file_name.replace('.txt', '')
        with open(f'{input_path}/{input_file_name}') as input_file, \
             open(f'{output_path}/{output_file_name}_specific_mask.json', 'w+') as output_file:
             for line in input_file.readlines():
                 text, target = line.strip().split('\t')
                 output_file.write(json.dumps({'text': text, 'target': target}))
                 output_file.write('\n')

def lot_prepare_training():
    from os import listdir
    from os.path import isfile, join

    input_path = 'data/LeapOfThought/mask-filling'
    output_path = 'data/LeapOfThought/fine-tuning'
    all_input_files = [f for f in listdir(input_path) if isfile(join(input_path, f)) if '3k' not in f]

    for input_file_name in all_input_files:
        output_file_name = input_file_name.replace('.txt', '.json')
        with open(f'{input_path}/{input_file_name}') as input_file, \
             open(f'{output_path}/{output_file_name}', 'w+') as output_file:
             for line in input_file.readlines():
                 text, target = line.strip().split('\t')
                 output_line = text.replace('<MASK>', target)
                 output_file.write(json.dumps({'text': output_line}))
                 output_file.write('\n')

def lot_prepare_training_specific_mask():
    from os import listdir
    from os.path import isfile, join

    input_path = 'data/LeapOfThought/mask-filling'
    output_path = 'data/LeapOfThought/fine-tuning'
    all_input_files = [f for f in listdir(input_path) if isfile(join(input_path, f)) if '3k' not in f]

    for input_file_name in all_input_files:
        output_file_name = input_file_name.replace('.txt', '')
        with open(f'{input_path}/{input_file_name}') as input_file, \
             open(f'{output_path}/{output_file_name}_specific_mask.json', 'w+') as output_file:
             for line in input_file.readlines():
                 text, target = line.strip().split('\t')
                 output_file.write(json.dumps({'text': text, 'target': target}))
                 output_file.write('\n')

def check_files():
    from os import listdir
    from os.path import isfile, join

    path = 'data/WikiData/mask-filling'
    all_files = [f for f in listdir(path) if isfile(join(path, f))]

    for file_name in all_files:
        if 'all' in file_name:
            with open(f'{path}/{file_name}') as f:
                for line in f.readlines():
                    if line.count('<MASK>') != 1:
                        print(f, line)

def classification_prepare():
    import gzip, json
    with gzip.open('data/LeapOfThought/dev.jsonl.gz') as input, \
         open('data/LeapOfThought/classification/dev.json', 'w+') as output:
        for line in input.readlines():
            instance = json.loads(line)
            phrase = instance['phrase']
            context = instance['context']
            output_text = f'{context} [SEP] {phrase}'
            output_dict = {'text': output_text}
            if instance['answer'] == 1:
                output_dict['label'] = 'True'
            else:
                output_dict['label'] = 'False'
            output.write(json.dumps(output_dict))
            output.write('\n')

def remove_pronoun():
    list_to_remove = ['it', 'he', 'him', 'she', 'her', 'you', 'i', 'me', 'we', 'us', 'they', 'them', 'this', 'these', 'those']
    with open('data/LeapOfThought/classification/lot_neutral_test.json') as input, \
         open('data/LeapOfThought/classification/lot_neutral_test_filtered.json', 'w+') as output:
        for line in input.readlines():
            instance = json.loads(line)
            text = instance['text']
            if any(substring in text.split() for substring in list_to_remove):
                continue
            else:
                output.write(json.dumps(instance))
                output.write('\n')

    with open('data/LeapOfThought/mask-filling/lot_neutral_test.txt') as input, \
         open('data/LeapOfThought/mask-filling/lot_neutral_test_filtered.txt', 'w+') as output:
        for line in input.readlines():
            text = line.split('\t')[0]
            if any(substring in text.split() for substring in list_to_remove):
                continue
            else:
                output.write(line)

    with open('data/WikiData/mask-filling/wd_neutral_test.txt') as input, \
         open('data/WikiData/mask-filling/wd_neutral_test_filtered.txt', 'w+') as output:
        for line in input.readlines():
            text = line.split('\t')[0]
            if any(substring in text.split() for substring in list_to_remove):
                continue
            else:
                output.write(line)

def retained_data_prepare():
    import ast
    with open('output/distilbert-base-uncased_lot_neutral_test_filtered.txt') as input, \
         open('data/LeapOfThought/mask-filling/lot_distilbert_retained.txt', 'w+') as output:
        for line in input.readlines():
            if len(line.strip().split('\t')) == 3:
                masked, target, results = line.strip().split('\t')
                results = ast.literal_eval(results)
                if target == results[0]['token_str']:
                    output.write(f'{masked}\t{target}\n')
    with open('output/distilbert-base-uncased_wd_neutral_test_filtered.txt') as input, \
         open('data/WikiData/mask-filling/wd_distilbert_retained.txt', 'w+') as output:
        for line in input.readlines():
            if len(line.strip().split('\t')) == 3:
                masked, target, results = line.strip().split('\t')
                results = ast.literal_eval(results)
                if target == results[0]['token_str']:
                    output.write(f'{masked}\t{target}\n')

def main():
    retained_data_prepare()

if __name__ == '__main__':
    main()