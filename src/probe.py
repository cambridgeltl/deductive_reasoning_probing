import gzip
import pickle
import pandas as pd
import json

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_entity_mapping(entities, save_dict=False, name=None):
    id_to_term = {}
    term_to_id = {}
    for entity in entities:
        entity = entity.strip()
        id = entity.split('\t')[0]
        terms = entity.split('\t')[1:]
        id_to_term[id] = terms
        for term in terms:
            term_to_id[term] = id
    if save_dict:
        save_obj(id_to_term, f'{name}_entity_id_to_term')
        save_obj(term_to_id, f'{name}_entity_term_to_id')
    return ({'id_to_term': id_to_term,
             'term_to_id': term_to_id})

def get_relation_mapping(relations, save_dict=False, name=None):
    id_to_term = {}
    term_to_id = {}
    for relation in relations:
        relation = relation.strip()
        id = relation.split('\t')[0]
        terms = relation.split('\t')[1:]
        id_to_term[id] = terms
        for term in terms:
            term_to_id[term] = id
    if save_dict:
        save_obj(id_to_term, f'{name}_relation_id_to_term')
        save_obj(term_to_id, f'{name}_relation_term_to_id')
    return ({'id_to_term': id_to_term,
             'term_to_id': term_to_id})

def get_entity_to_related_entities(triples, save_dict=False, name=None):
    entity_to_related_entities = {}
    for _, row in triples.iterrows():
        entity1, relation, entity2 = row['entity1'], row['relation'], row['entity2']
        if entity1 != entity2:
            entity_to_related_entities.setdefault(entity1,{}).setdefault(entity2,{})[relation] = 'subj'
            entity_to_related_entities.setdefault(entity2,{}).setdefault(entity1,{})[relation] = 'obj'
        else:
            print(f'Warning! Skipped {row}!')
    if save_dict:
        save_obj(entity_to_related_entities, f'{name}_entity_to_related_entities')
    return entity_to_related_entities

def get_entity_to_triples(data, entity_to_related_entities, save_dict=False, name=None):
    entity1_to_triples = {}
    entity2_to_triples = {}
    for _, row in data.iterrows():
        entity1, relation, entity2 = row['entity1'], row['relation'], row['entity2']
        if entity1 in entity_to_related_entities:
            entity2_to_triples.setdefault(entity2,{}).setdefault(relation,set()).add(entity1)
        if entity2 in entity_to_related_entities:
            entity1_to_triples.setdefault(entity1,{}).setdefault(relation,set()).add(entity2)
    if save_dict:
        save_obj(entity1_to_triples, f'{name}_entity1_to_triples')
        save_obj(entity2_to_triples, f'{name}_entity2_to_triples')
    return ({'entity1_to_triples': entity1_to_triples,
            'entity2_to_triples': entity2_to_triples})

def create_positive_data(data, save_path, entity_to_related_entities, name):
    entity1_to_triples = {}
    entity2_to_triples = {}
    for _, row in data.iterrows():
        entity1, relation, entity2 = row['entity1'], row['relation'], row['entity2']
        if entity1 in entity_to_related_entities:
            entity2_to_triples.setdefault(entity2,{}).setdefault(relation,set()).add(entity1)
            for related_entity, hierarchical_relations in entity_to_related_entities[entity1].items():
                if related_entity in entity2_to_triples[entity2][relation]:
                    for hierarchical_relation in hierarchical_relations:
                        with open(f'{save_path}/{name}_{hierarchical_relation}.txt', 'a') as f:
                            f.write(f'{related_entity}\t{relation}\t{entity2},{entity1}\t{relation}\t{entity2},')
                            if hierarchical_relations[hierarchical_relation] == 'subj':
                                f.write(f'{entity1}\t{hierarchical_relation}\t{related_entity}\n')
                            elif hierarchical_relations[hierarchical_relation] == 'obj':
                                f.write(f'{related_entity}\t{hierarchical_relation}\t{entity1}\n')
        if entity2 in entity_to_related_entities:
            entity1_to_triples.setdefault(entity1,{}).setdefault(relation,set()).add(entity2)
            for related_entity, hierarchical_relations in entity_to_related_entities[entity2].items():
                if related_entity in entity1_to_triples[entity1][relation]:
                    for hierarchical_relation in hierarchical_relations:
                        with open(f'{save_path}/{name}_{hierarchical_relation}.txt', 'a') as f:
                            f.write(f'{entity1}\t{relation}\t{related_entity},{entity1}\t{relation}\t{entity2},')
                            if hierarchical_relations[hierarchical_relation] == 'subj':
                                f.write(f'{entity2}\t{hierarchical_relation}\t{related_entity}\n')
                            elif hierarchical_relations[hierarchical_relation] == 'obj':
                                f.write(f'{related_entity}\t{hierarchical_relation}\t{entity2}\n')
            
def prepare(data_path, save_path, relation_path=None, filtered_id_to_terms=None, save_dict=False, name=None):
    if relation_path:
        with open(relation_path, 'r') as f:
            relations = f.readlines()
            output = get_relation_mapping(relations, save_dict=save_dict, name=name)
        id_to_term = output['id_to_term']
    elif filtered_id_to_terms:
        id_to_term = filtered_id_to_terms
    data = pd.read_csv(data_path, sep='\t', names=['entity1', 'relation', 'entity2'])
    hierarchical_triples = data[data['relation'].isin(id_to_term)]
    hierarchical_entity_to_related_entities = get_entity_to_related_entities(hierarchical_triples, save_dict=save_dict, name=name)
    data = data[data['entity1'].isin(hierarchical_entity_to_related_entities) | data['entity2'].isin(hierarchical_entity_to_related_entities)]
    create_positive_data(data=data, save_path=save_path, entity_to_related_entities=hierarchical_entity_to_related_entities, name=name)

def triple_id_to_term(triple, entity_id_to_term, relation_id_to_term, n_tokens, vocab_list, required_entities):
    entity1, relation, entity2 = triple.split('\t')
    try:
        if entity1 in required_entities:
            entity1_list = [text.lower() for text in entity_id_to_term[entity1] \
                            if all([token in vocab_list for token in text.lower().split()])]
            entity1_text = [text for text in entity1_list if len(text.split()) == n_tokens]
            entity1_text = entity1_text[0]
        else:
            entity1_text = entity_id_to_term[entity1][0].lower()
        # keep the relation id
        relation_text = relation

        if entity2 in required_entities:
            entity2_list = [text.lower() for text in entity_id_to_term[entity2] \
                            if all([token in vocab_list for token in text.lower().split()])]
            entity2_text = [text for text in entity2_list if len(text.split()) == n_tokens]                
            entity2_text = entity2_text[0]
        else:
            entity2_text = entity_id_to_term[entity2][0].lower()

        return(f'{entity1_text}\t{relation_text}\t{entity2_text}')
    except:
        return None

def create_rdf(data_path, entity_id_to_term, relation_id_to_term, vocab, n_tokens):
    models_mapping = {'bert':'bert-base-uncased'}
    if isinstance(vocab, set):
        vocab_list = vocab
    elif isinstance(vocab, str):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(models_mapping[vocab])
        vocab_list = tokenizer.get_vocab().keys()
    with open(data_path, 'r') as f:
        lines = f.readlines()
        filename = data_path.split('/')[-1].split('.')[0]
        with open(f'data/multi-token/{n_tokens}/{filename}_{n_tokens}_rdf.json', 'w+') as e:
            for line in lines:
                output = {}
                triple3 = line.strip().split(',')[2]
                required_entities = [triple3.split('\t')[0], triple3.split('\t')[2]]
                for i, triple in enumerate(line.strip().split(',')):
                    output[f'triple{i+1}'] = triple_id_to_term(triple, entity_id_to_term, relation_id_to_term, \
                                                               n_tokens, vocab_list, required_entities)
                if not output or None in output.values():
                    continue
                else:
                    e.write(f'{json.dumps(output)}\n')

def rdf_to_text(line, prompt_dict, entity_to_mask):
    output = []
    triples = json.loads(line.strip())
    for key in triples:
        triple = triples[key]
        entity1, id, entity2 = triple.strip().split('\t')
        if id in prompt_dict:
            if key == 'triple3':
                output.append(prompt_dict[id].replace('[X]', entity1).replace('[Y]', entity2))
            elif entity_to_mask == entity1:
                output.append(prompt_dict[id].replace('[X]', '<MASK>').replace('[Y]', entity2))
            elif entity_to_mask == entity2:
                output.append(prompt_dict[id].replace('[X]', entity1).replace('[Y]', '<MASK>'))
            else:
                output.append(prompt_dict[id].replace('[X]', entity1).replace('[Y]', entity2))
        else:
            return None
    return output

def rdfs_to_text(data_path, prompt_path):
    prompt_dict = {}
    with open(prompt_path) as f:
        for line in f.readlines():
            id, prompt = line.split('\t')
            prompt_dict[id] = prompt.strip()
    with open(data_path) as f:
        lines = f.readlines()
        filename = data_path.split('_rdf.json')[0]
        with open(f'{filename}_text.csv', 'w+') as e:
            for line in lines:
                output = rdf_to_text(line, prompt_dict)
                if output:
                    e.write('\t'.join(output) + '\n')

def top_relations(n_tokens):
    with open(f'data/multi-token/{n_tokens}/all_{n_tokens}_rdf.json') as f:
        rdfs = f.readlines()
    relation_id_to_term = load_obj('all_relation_id_to_term')
    ids = {}
    for id in relation_id_to_term.keys():
        query = f'\\t{id}\\t'
        if query in ' '.join(rdfs):
            ids[id] = ' '.join(rdfs).count(query)
    ids = sorted(ids, key=ids.get, reverse=True)[:20]
    with open('data/wikidata5m/wikidata5m_alias/wikidata5m_relation.txt') as f:
        relations = f.readlines()
    with open(f'data/multi-token/{n_tokens}/top_relations.txt', 'w+') as f:
        for relation in relations:
            if relation.split('\t')[0] in ids:
                f.write(relation)

def top_hierarchical_triples(n_tokens):
    with open(f'data/multi-token/{n_tokens}/all_{n_tokens}_rdf.json') as f:
        rdfs = f.readlines()
    hierarchical_triples = {}
    for line in rdfs:
        triple3 = json.loads(line)['triple3']
        hierarchical_triples[triple3] = hierarchical_triples.get(triple3, 0) + 1
    hierarchical_triples = sorted(hierarchical_triples, key=hierarchical_triples.get, reverse=True)[:50]
    with open(f'data/multi-token/{n_tokens}/top_hierarchical_triples.txt', 'w+') as f:
        for triples in hierarchical_triples:
            f.write(f'{triples}\n')

def generate_random_sample(n_tokens, sample_size):
    SEED = 0
    from sklearn.model_selection import train_test_split
    import random

    random.seed(SEED)

    prompt = pd.read_csv('data/multi-token/prompt.txt', sep='\t')
    top_relations = prompt['id'].tolist()

    sample_triples_train = {} 
    sample_triples_valid = {}
    sample_triples_test = {}

    with open(f'data/multi-token/{n_tokens}/top_hierarchical_triples.txt') as f:
        top_hierarchical_triples = f.readlines()

    top_hierarchical_triples = [triple.strip() for triple in top_hierarchical_triples]
    top_hierarchical_triples_train, top_hierarchical_triples_valid = train_test_split(top_hierarchical_triples, test_size = 0.33, random_state=SEED)
    top_hierarchical_triples_valid, top_hierarchical_triples_test = train_test_split(top_hierarchical_triples_valid, test_size = 0.5, random_state=SEED)

    with open(f'data/multi-token/{n_tokens}/all_{n_tokens}_rdf.json') as f:    
        for line in f.readlines():
            triple1 = json.loads(line)['triple1']
            relation1 = triple1.split('\t')[1]
            triple2 = json.loads(line)['triple2']
            relation2 = triple2.split('\t')[1]
            triple3 = json.loads(line)['triple3']
            if relation1 in top_relations and relation2 in top_relations:
                if triple3 in top_hierarchical_triples_train:
                    sample_triples_train.setdefault(triple3, []).append(line.strip())
                elif triple3 in top_hierarchical_triples_valid:
                    sample_triples_valid.setdefault(triple3, []).append(line.strip())
                elif triple3 in top_hierarchical_triples_test:
                    sample_triples_test.setdefault(triple3, []).append(line.strip())

    with open(f'data/multi-token/{n_tokens}/train.json','w+') as f:
        for key in sample_triples_train:
            if len(sample_triples_train[key]) < sample_size:
                random_samples = sample_triples_train[key]
            else:
                random_samples = random.sample(sample_triples_train[key], sample_size)
            for sample in random_samples:
                f.write(sample + '\n')
    with open(f'data/multi-token/{n_tokens}/valid.json','w+') as f:
        for key in sample_triples_valid:
            if len(sample_triples_valid[key]) < sample_size:
                random_samples = sample_triples_valid[key]
            else:
                random_samples = random.sample(sample_triples_valid[key], sample_size)
            for sample in random_samples:
                f.write(sample + '\n')
    with open(f'data/multi-token/{n_tokens}/test.json','w+') as f:
        for key in sample_triples_test:
            if len(sample_triples_test[key]) < sample_size:
                random_samples = sample_triples_test[key]
            else:
                random_samples = random.sample(sample_triples_test[key], sample_size)
            for sample in random_samples:
                f.write(sample + '\n')

def direction(n_tokens):
    with open(f'data/multi-token/{n_tokens}/samples.json') as f:
        samples = f.readlines()
    for sample in samples:
        sample = json.loads(sample)
        triple1 = sample['triple1']
        triple2 = sample['triple2']
        triple3 = sample['triple3']
        entity1 = triple3.split('\t')[0]
        entity2 = triple3.split('\t')[2]
        if entity1 in triple1.split('\t'):
            print('1->2')
        else:
            print('2->1')

def prepare_masked_text(n_tokens, prompt=False, split='valid'):
    prompt_dict = {}
    with open('data/multi-token/prompt.txt') as f:
        for line in f.readlines():
            id, prompt = line.split('\t')
            prompt_dict[id] = prompt.strip()

    with open(f'data/multi-token/{n_tokens}/{split}.json', encoding='utf-8') as f:
        with open(f'data/multi-token/{n_tokens}/{split}_forward.txt','w+', encoding='utf-8') as forward, \
             open(f'data/multi-token/{n_tokens}/{split}_forward_explicit.txt','w+', encoding='utf-8') as forward_explicit, \
             open(f'data/multi-token/{n_tokens}/{split}_backward.txt','w+', encoding='utf-8') as backward, \
             open(f'data/multi-token/{n_tokens}/{split}_backward_explicit.txt','w+', encoding='utf-8') as backward_explicit:
            lines = f.readlines()
            for line in lines:
                triple_dict = json.loads(line.strip())
                triple1 = triple_dict['triple1']
                triple2 = triple_dict['triple2']
                triple3 = triple_dict['triple3']

                entity1 = triple3.split('\t')[0]
                entity2 = triple3.split('\t')[2]

                if entity1 == entity2:
                    continue

                if entity1 in triple1.split('\t'):
                    text1, text2, text3 = rdf_to_text(line, prompt_dict, entity2)

                    if prompt:
                        forward.write(f'if {text1} then {text2}\t{entity2}\n')
                        forward_explicit.write(f'if {text1} and {text3} then {text2}\t{entity2}\n')
                    else:
                        forward.write(f'{text1} {text2}\t{entity2}\n')
                        forward_explicit.write(f'{text1} {text3} {text2}\t{entity2}\n')
                    
                    text1, text2, text3 = rdf_to_text(line, prompt_dict, entity1)

                    if prompt:
                        backward.write(f'if {text2} then {text1}\t{entity1}\n')
                        backward_explicit.write(f'if {text2} and {text3} then {text1}\t{entity1}\n')
                    else:
                        backward.write(f'{text2} {text1}\t{entity1}\n')
                        backward_explicit.write(f'{text2} {text3} {text1}\t{entity1}\n')
                else:
                    text1, text2, text3 = rdf_to_text(line, prompt_dict, entity2)

                    if prompt:
                        forward.write(f'if {text2} then {text1}\t{entity2}\n')
                        forward_explicit.write(f'if {text2} and {text3} then {text1}\t{entity2}\n')
                    else:
                        forward.write(f'{text2} {text1}\t{entity2}\n')
                        forward_explicit.write(f'{text2} {text3} {text1}\t{entity2}\n')

                    text1, text2, text3 = rdf_to_text(line, prompt_dict, entity1)

                    if prompt:
                        backward.write(f'if {text1} then {text2}\t{entity1}\n')
                        backward_explicit.write(f'if {text1} and {text3} then {text2}\t{entity1}\n')
                    else:
                        backward.write(f'{text1} {text2}\t{entity1}\n')
                        backward_explicit.write(f'{text1} {text3} {text2}\t{entity1}\n')

def prepare_masked_text_wiki():
    prompt_dict = {}
    with open('data/multi-token/prompt.txt') as f:
        for line in f.readlines():
            id, prompt = line.split('\t')
            prompt_dict[id] = prompt.strip()

    with open(f'data/WikiData/mask-filling/all_1_rdf.json', encoding='utf-8') as input, \
         open(f'data/WikiData/mask-filling/all.txt','w+', encoding='utf-8') as mf_output:
            lines = input.readlines()
            for line in lines:
                triple_dict = json.loads(line.strip())
                triple1 = triple_dict['triple1']
                triple2 = triple_dict['triple2']
                triple3 = triple_dict['triple3']

                entity1 = triple3.split('\t')[0]
                entity2 = triple3.split('\t')[2]

                if entity1 == entity2:
                    continue

                if entity1 in triple1.split('\t'):
                    if rdf_to_text(line, prompt_dict, entity2):
                        text1, text2, text3 = rdf_to_text(line, prompt_dict, entity2)
                        mf_output.write(f'{text1} {text3} {text2}\t{entity2}\n')

                else:
                    if rdf_to_text(line, prompt_dict, entity2):
                        text1, text2, text3 = rdf_to_text(line, prompt_dict, entity2)
                        mf_output.write(f'{text2} {text3} {text1}\t{entity2}\n')


def eval(path, filename, model_name):
    from transformers import pipeline
    unmasker = pipeline("fill-mask", model=model_name, tokenizer=model_name, top_k=10, device=0)

    r_at_k = {}
    scores = []

    with open(f'{path}/{filename}') as f:
        lines = f.readlines()

    with open(f'output/{model_name}_{filename}','w+') as f:
        for line in lines:
            text, target = line.strip().split('\t')
            masked = text.replace('<MASK>', unmasker.tokenizer.mask_token)

            candidates = unmasker(masked)
            f.write(f'{text}\t{target}\t{candidates}\n')

            score = 0

            for i in candidates:
                if i['token_str'] == target:
                    score = i['score']

            scores.append(score)

            candidates = [candidate['token_str'].strip() for candidate in candidates]

            for k in range(1,11):
                if target in candidates[:k]:
                    r_at_k.setdefault(k, []).append(1)
                else:
                    r_at_k.setdefault(k, []).append(0)

        for k in r_at_k:
            average_recall = sum(r_at_k[k])/len(r_at_k[k])
            average_recall = round(average_recall, 4)
            f.write(f'{k}\t{average_recall}\n')
        f.write('\n')
        average_score = sum(scores)/len(scores)
        average_score = round(average_score, 4)
        f.write(f'{average_score}\n')
        f.write('\n')

def classification_eval(path, filename, model_name):
    from transformers import pipeline
    classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, device=0)

    count = []

    with open(f'{path}/{filename}') as f:
        lines = f.readlines()

    with open(f'output/classification/{model_name}_{filename}','w+') as f:
        for line in lines:
            instance = json.loads(line)
            text = instance['text']
            label = instance['label']
            predicted_label = classifier(text)[0]['label']
            f.write(f'{text}\t{label}\t{predicted_label}\n')

            if label == predicted_label:
                count.append(1)
            else:
                count.append(0)

        accuracy = sum(count) / len(count)
        f.write(f'{accuracy}\n')
        f.write('\n')

def split(n_tokens):
    SEED = 0
    from sklearn.model_selection import train_test_split

    with open(f'data/multi-token/{n_tokens}/samples.json') as f:
        data = f.readlines()
            
    train, valid = train_test_split(data, test_size = 0.33, random_state=SEED)
    with open(f'data/multi-token/{n_tokens}/train.json', 'w+') as f_train:
        for line in train:
            f_train.write(line)
    with open(f'data/multi-token/{n_tokens}/valid.json', 'w+') as f_valid:
        for line in valid:
            f_valid.write(line)

def check_unprompted_relation():
    prompt = pd.read_csv('data/multi-token/prompt.txt', sep='\t')
    relations = prompt['id'].tolist()
    for n in range(1,6):
        with open(f'data/multi-token/{n}/samples.json') as f:
            samples = f.readlines()
        unlabeled = set()
        for sample in samples:
            sample = json.loads(sample)
            for triple in sample.values():
                relation = triple.split('\t')[1]
                if relation not in relations:
                    unlabeled.add(relation)
        print(n, unlabeled)

def read_vocab():
    with open('data/multi-token/vocab.txt') as f:
        vocab_list = [vocab[:-1] for vocab in f.readlines()]
    return set(vocab_list)

def multi_token_prepare():
    vocab_list = read_vocab()
    entity_id_to_term = load_obj('all_entity_id_to_term')
    for n in range(1,6):
        create_rdf(data_path='data/multi-token/all.txt', entity_id_to_term=entity_id_to_term, \
                   relation_id_to_term=None, vocab=vocab_list, n_tokens=n)
        top_relations(n_tokens=n)
        top_hierarchical_triples(n_tokens=n)
        generate_random_sample(n_tokens=n, sample_size=50)
        # split(n_tokens=n)

def prepare_all():
    for split in ['train', 'valid', 'test']:
        for n in range(1,6):
            prepare_masked_text(n_tokens=n, prompt=True, split=split)

def lot_preprocessing(split):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_list = tokenizer.get_vocab().keys()

    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    def get_response(input_text,num_return_sequences,num_beams):
        batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def get_paraphrase(context):
        paraphrase = ' '.join([get_response(sentence, 10, 10)[0] for sentence in context.split('.') if sentence])
        return paraphrase

    with gzip.open(f'data/LeapOfThought/{split}.jsonl.gz', 'rt', encoding='UTF-8') as zipfile:
        input_lines = zipfile.readlines()
    with open(f'data/LeapOfThought/fine-tuning/lot_{split}.json', 'w+') as ft_output, \
         open(f'data/LeapOfThought/fine-tuning/lot_pegasus_para_{split}.json', 'w+') as ft_para_output, \
         open(f'data/LeapOfThought/mask-filling/lot_{split}.txt', 'w+') as mf_output, \
         open(f'data/LeapOfThought/mask-filling/lot_pegasus_para_{split}.txt', 'w+') as mf_para_output:
        for line in input_lines:
            my_object = json.loads(line)
            if my_object['answer'] == 1:
                subj = my_object['metadata']['statement']['subject'].strip()
                context = my_object['context'].strip()
                phrase = my_object['phrase'].strip()
                target = subj
                if target in vocab_list:
                        ft_json = {'text': f'{context} {phrase}'}
                        ft_output.write(f'{json.dumps(ft_json)}\n')
                        masked_text = context + ' ' + phrase.replace(target, '<MASK>')
                        mf_output.write(f'{masked_text}\t{target}\n')

                        context = get_paraphrase(context)

                        ft_json = {'text': f'{context} {phrase}'}
                        ft_para_output.write(f'{json.dumps(ft_json)}\n')
                        masked_text = context + ' ' + phrase.replace(target, '<MASK>')
                        mf_para_output.write(f'{masked_text}\t{target}\n')

def lot_classification_preprocessing(split):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_list = tokenizer.get_vocab().keys()

    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    def get_response(input_text,num_return_sequences,num_beams):
        batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def get_paraphrase(context):
        paraphrase = ' '.join([get_response(sentence, 10, 10)[0] for sentence in context.split('.') if sentence])
        return paraphrase

    with open(f'data/LeapOfThought/classification/lot_{split}.json') as input, \
         open(f'data/LeapOfThought/classification/lot_pegasus_para_{split}.json', 'w+') as output:
         for line in input.readlines():
            instance = json.loads(line)
            text = instance['text']
            label = instance['label']
            context, conclusion = text.split('[SEP]')
            context = get_paraphrase(context.strip())
            conclusion = get_paraphrase(conclusion)
            text = f'{context} [SEP] {conclusion}'
            output.write(json.dumps({'text': text, 'label': label}))
            output.write('\n')

def lot_prepare_all():
    for split in ['train', 'dev', 'test']:
        lot_preprocessing(split)

def pp_preprocessing(split):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_list = tokenizer.get_vocab().keys()

    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    def get_response(input_text,num_return_sequences,num_beams):
        batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def get_paraphrase(context):
        paraphrase = ' '.join([get_response(sentence, 10, 10)[0] for sentence in context.split('.') if sentence])
        return paraphrase

    path = 'data/ParaPattern'
    with open(f'{path}/{split}.source') as source, \
         open(f'{path}/{split}.target') as target:
         source_lines = source.readlines()
         target_lines = target.readlines()
    
    with open(f'{path}/fine-tuning/pp_{split}.json', 'w+') as ft_output, \
         open(f'{path}/fine-tuning/pp_pegasus_para_{split}.json', 'w+') as ft_para_output, \
         open(f'{path}/mask-filling/pp_{split}.txt', 'w+') as mf_output, \
         open(f'{path}/mask-filling/pp_pegasus_para_{split}.txt', 'w+') as mf_para_output:
        for line in list(zip(source_lines, target_lines)):
            context = line[0].strip()
            phrase = line[1].strip()
            target_candidates = [' '.join([word for word in phrase.replace('.', ' ').split() if word not in context_sent.split()]) for context_sent in context.split('.') if context_sent]
            for target in target_candidates:
                if target in vocab_list:
                    ft_json = {'text': f'{context} {phrase}'}
                    ft_output.write(f'{json.dumps(ft_json)}\n')
                    masked_text = context + ' ' + phrase.replace(target, '<MASK>')
                    mf_output.write(f'{masked_text}\t{target}\n')

                    context = get_paraphrase(context)

                    ft_json = {'text': f'{context} {phrase}'}
                    ft_para_output.write(f'{json.dumps(ft_json)}\n')
                    masked_text = context + ' ' + phrase.replace(target, '<MASK>')
                    mf_para_output.write(f'{masked_text}\t{target}\n')

def pp_prepare_all():
    for split in ['train', 'dev']:
        pp_preprocessing(split)

def wiki_paraphrase():
    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    def get_response(input_text,num_return_sequences,num_beams):
        batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def get_paraphrase(context):
        paraphrase = ' '.join([get_response(sentence, 10, 10)[0] if '<MASK>' not in sentence else sentence for sentence in context.split('.') if sentence])
        return paraphrase

    path = 'data/WikiData'

    with open(f'{path}/mask-filling/all.txt') as input:
        lines = input.readlines()

    with open(f'{path}/fine-tuning/wd_all.json', 'w+') as ft_output, \
         open(f'{path}/fine-tuning/wd_pegasus_para_all.json', 'w+') as ft_para_output, \
         open(f'{path}/mask-filling/wd_all.txt', 'w+') as mf_output, \
         open(f'{path}/mask-filling/wd_pegasus_para_all.txt', 'w+') as mf_para_output:
        for line in lines:
            context, target = line.strip().split('\t')
            mf_output.write(f'{context}\t{target}\n')

            unmasked_context = context.replace('<MASK>', target)
            ft_json = {'text': f'{unmasked_context}'}
            ft_output.write(f'{json.dumps(ft_json)}\n')

            context = get_paraphrase(context)
            context = context + '.'
            unmasked_context = context.replace('<MASK>', target)
            ft_json = {'text': f'{unmasked_context}'}
            ft_para_output.write(f'{json.dumps(ft_json)}\n')

            mf_para_output.write(f'{context}\t{target}\n')

def txt_to_json(input_path, output_path):
    with open(input_path) as txt, \
         open(output_path, 'w+') as j:
        for line in txt.readlines():
            text, target = line.strip().split('\t')
            unmasked_text = text.replace('<MASK>', target)
            output_json = {'text': f'{unmasked_text}'}
            j.write(f'{json.dumps(output_json)}\n')

def lot_txt_to_json():
    path = 'data/LeapOfThought'

    splits = ['test', 'dev', 'train']

    files = ['lot_nla_para', 'lot_nla_syn', 'lot_pred_negation']

    for file in files:
        for split in splits:
            txt_to_json(f'{path}/mask-filling/{file}_{split}.txt', f'{path}/fine-tuning/{file}_{split}.json')

def eval_all():
    split = 'test'

    path = 'data/multi-token/1'

    texts = [f'{split}_forward.txt',
             f'{split}_forward_explicit.txt',
             f'{split}_backward.txt',
             f'{split}_backward_explicit.txt']

    model_names = ['bert-base-uncased',
                   'distilbert-base-uncased',
                   'bert-large-uncased',
                   'models/forward', 
                   'models/forward_explicit']

    for model_name in model_names:
        for text in texts:
            eval(path, text, model_name)

def eval_lot():
    split = 'test'

    path = 'data/LeapOfThought/mask-filling'

    texts = [f'lot_{split}.txt',
             f'lot_pegasus_para_{split}.txt',
             f'lot_nla_para_{split}.txt',
             f'lot_nla_syn_{split}.txt',
            #  f'lot_nla_syn_det_{split}.txt',
            #  f'lot_neutral_{split}.txt',
             f'lot_pred_negation_{split}_correct.txt',
            #  f'lot_no_periods_{split}.txt',
            #  f'lot_random_order_{split}.txt',
            #  f'lot_random_order_fix_{split}.txt',
            #  f'lot_neutral_{split}_filtered.txt',
             f'lot_retained.txt'
            ]

    model_names = [#'bert-base-uncased',
                #    'models/lot',
                #    'models/lot_specific',
                #    'models/lot_pegasus_para',
                #    'models/lot_nla_para',
                #    'models/lot_nla_syn',
                #    'models/lot_nla_syn_det',
                #    'models/lot_all',
                #    'models/lot_bert_large',
                #    'models/lot_distilbert',
                #    'models/lot_specific_bert_large',
                #    'models/lot_specific_distilbert',
                #    'distilbert-base-uncased',
                #    'bert-large-uncased',
                   'models/lot_bert',
                   'models/lot_specific_bert',
                   ]

    for model_name in model_names:
        for text in texts:
            eval(path, text, model_name)

def eval_pp():
    split = 'dev'

    path = 'data/ParaPattern/mask-filling'

    texts = [f'pp_{split}.txt',
             f'pp_pegasus_para_{split}.txt',]

    model_names = ['bert-base-uncased']

    # model_names = ['bert-base-uncased',
    #                'models/pp',
    #                'models/pp_pegasus_para',]

    for model_name in model_names:
        for text in texts:
            eval(path, text, model_name)

def eval_wd():
    split = 'test'
    path = 'data/WikiData/mask-filling'
    model_names = [#'bert-base-uncased',
                    # 'models/wd',
                    # 'models/wd_specific',
                #    'models/wd_pegasus_para',
                #    'models/wd_nla_para',
                #    'models/wd_nla_syn',
                #    'models/wd_nla_syn_det',
                #    'models/wd_all',
                #    'models/wd_bert_large',
                #    'models/wd_distilbert',
                #    'models/wd_specific_bert_large',
                #    'models/wd_specific_distilbert',
                #    'distilbert-base-uncased',
                #    'bert-large-uncased',
                    'models/wd_bert',
                    'models/wd_specific_bert'
                   ]


    for model_name in model_names:
        for max_num in [50]:
            texts = [f'wd_{split}_{max_num}.txt',
                     f'wd_pegasus_para_{split}_{max_num}.txt',
                     f'wd_nla_para_{split}_{max_num}.txt',
                     f'wd_nla_syn_{split}_{max_num}.txt',
                    #  f'wd_nla_syn_det_{split}_{max_num}.txt',
                     f'wd_pred_negation_{split}_{max_num}_correct.txt',
                    #  f'wd_neutral_{split}.txt',
                    #  f'wd_no_periods_{split}_{max_num}.txt',
                    #  f'wd_random_order_{split}_{max_num}.txt',
                    #  f'wd_random_order_fix_{split}_{max_num}.txt',
                    #  f'wd_neutral_{split}_filtered.txt',
                     f'wd_retained.txt',
                    ]
            for text in texts:
                eval(path, text, model_name)

def eval_classification_all():
    split = 'test'
    path = 'data/LeapOfThought/classification'
    model_names = ['models/lot_classifier']

    texts = [#f'lot_{split}.json',
            #  f'lot_nla_para_{split}.json',
            #  f'lot_nla_syn_{split}.json',
            #  f'lot_pred_negation_{split}.json',
             f'lot_pegasus_para_{split}.json',]
            #  f'lot_neutral_{split}.json',
            #  f'lot_random_order_fix_{split}.json',
            #  f'lot_neutral_{split}_filtered.json',]

    for model_name in model_names:
        for text in texts:
            classification_eval(path, text, model_name)

def main():
    # lot_classification_preprocessing('test')
    # eval_classification_all()
    eval_lot()
    eval_wd()

if __name__ == '__main__':
    main()