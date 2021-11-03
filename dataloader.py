#! -*- coding:utf-8 -*-

import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from keras.preprocessing.sequence import pad_sequences

import log


def find_entity(source: List[int], target: List[int]) -> int:
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def to_tuple(sent: str):
    """ list to tuple (inplace operation)
    """
    triple_list = []
    for triple in sent['triple_list']:
        triple_list.append(tuple(triple))
    sent['triple_list'] = triple_list


def filter_data(fpath: str, rel2id: Dict):
    filtered_data = []
    for obj in json.load(open(fpath)):
        filtered_triples = []
        if 'NYT11-HRL' in fpath and len(obj['triple_list']) != 1:
            continue
        for triple in obj['triple_list']:
            if triple[1] not in rel2id:
                continue
            filtered_triples.append(triple)
        if not filtered_triples:
            continue
        obj['triple_list'] = filtered_triples
        filtered_data.append(obj)
    return filtered_data


def load_rel(rel_path: str) -> Tuple[Dict, Dict, List, int]:
    id2rel, rel2id = json.load(open(rel_path))
    all_rels = list(id2rel.keys())
    id2rel = {int(i): j for i, j in id2rel.items()}
    return id2rel, rel2id, all_rels


def load_data(fpath: str, rel2id: Dict, is_train: bool = False) -> List:
    data = filter_data(fpath, rel2id)
    if is_train:
        text_lens = [len(obj['text'].split()) for obj in data]
        log.info("train text insight")
        log.info(f" max len: {max(text_lens)}")
        log.info(f" min len: {min(text_lens)}")
        log.info(f" avg len: {sum(text_lens) / len(text_lens)}")
    for sent in data:
        to_tuple(sent)
    log.info(f"data len: {len(data)}")
    return data


class DataGenerator:
    def __init__(self, datas: List, tokenizer: object, rel2id: Dict, all_rels: List, max_len: int,
                 batch_size: int = 32, max_sample_triples: Optional[int] = None, neg_samples: Optional[int] = None):
        self.max_sample_triples = max_sample_triples
        self.neg_samples = neg_samples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rel2id = rel2id
        self.rels_set = list(rel2id.values())
        self.relation_size = len(rel2id)
        self.num_rels = len(all_rels)
        self.all_rels = all_rels

        self.datas = []

        for data in datas:
            pos_datas = []
            neg_datas = []

            text_tokened = tokenizer.encode(data['text'])
            entity_set = set()  # (head idx, tail idx)
            triples_set = set()   # (sub head, sub tail, obj head, obj tail, rel)
            subj_rel_set = set()   # (sub head, sub tail, rel)
            subj_set = set()   # (sub head, sub tail)
            rel_set = set()
            trans_map = defaultdict(list)   # {(sub_head, rel): [tail_heads]}
            for triple in data['triple_list']:
                subj, rel, obj = triple
                rel_idx = self.rel2id[rel]
                subj_tokened = tokenizer.encode(subj)
                obj_tokened = tokenizer.encode(obj)
                subj_head_idx = find_entity(text_tokened.ids, subj_tokened.ids[1:-1])
                subj_tail_idx = subj_head_idx + len(subj_tokened.ids[1:-1]) - 1
                obj_head_idx = find_entity(text_tokened.ids, obj_tokened.ids[1:-1])
                obj_tail_idx = obj_head_idx + len(obj_tokened.ids[1:-1]) - 1
                if subj_head_idx == -1 or obj_head_idx == -1:
                    continue
                entity_set.add((subj_head_idx, subj_tail_idx, 0))
                entity_set.add((obj_head_idx, obj_tail_idx, 1))
                subj_rel_set.add((subj_head_idx, subj_tail_idx, rel_idx))
                subj_set.add((subj_head_idx, subj_tail_idx))
                triples_set.add(
                    (subj_head_idx, subj_tail_idx, obj_head_idx, obj_tail_idx, rel_idx)
                )
                rel_set.add(rel_idx)
                trans_map[(subj_head_idx, subj_tail_idx, rel_idx)].append(obj_head_idx)

            if not rel_set:
                continue

            entity_heads = np.zeros((self.max_len, 2))
            entity_tails = np.zeros((self.max_len, 2))
            for (head, tail, _type) in entity_set:
                entity_heads[head][_type] = 1
                entity_tails[tail][_type] = 1

            rels = np.zeros(self.relation_size)
            for idx in rel_set:
                rels[idx] = 1

            if self.max_sample_triples is not None:
                triples_list = list(triples_set)
                np.random.shuffle(triples_list)
                triples_list = triples_list[:self.max_sample_triples]
            else:
                triples_list = list(triples_set)

            neg_history = set()
            for subj_head_idx, subj_tail_idx, obj_head_idx, obj_tail_idx, rel_idx in triples_list:
                current_neg_datas = []
                sample_obj_heads = np.zeros(self.max_len)
                for idx in trans_map[(subj_head_idx, subj_tail_idx, rel_idx)]:
                    sample_obj_heads[idx] = 1.0
                # postive samples
                pos_datas.append({
                    'token_ids': text_tokened.ids,
                    'segment_ids': text_tokened.type_ids,
                    'entity_heads': entity_heads,
                    'entity_tails': entity_tails,
                    'rels': rels,
                    'sample_subj_head': subj_head_idx,
                    'sample_subj_tail': subj_tail_idx,
                    'sample_rel': rel_idx,
                    'sample_obj_heads': sample_obj_heads,
                })

                # 1. inverse (tail as subj)
                neg_subj_head_idx = obj_head_idx
                neg_sub_tail_idx = obj_tail_idx
                neg_pair = (neg_subj_head_idx, neg_sub_tail_idx, rel_idx)
                if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                    current_neg_datas.append({
                        'token_ids': text_tokened.ids,
                        'segment_ids': text_tokened.type_ids,
                        'entity_heads': entity_heads,
                        'entity_tails': entity_tails,
                        'rels': rels,
                        'sample_subj_head': neg_subj_head_idx,
                        'sample_subj_tail': neg_sub_tail_idx,
                        'sample_rel': rel_idx,
                        'sample_obj_heads': np.zeros(self.max_len),  # set 0 for negative samples
                    })
                    neg_history.add(neg_pair)

                # 2. (pos sub, neg_rel)
                for neg_rel_idx in rel_set - {rel_idx}:
                    neg_pair = (subj_head_idx, subj_tail_idx, neg_rel_idx)
                    if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                        current_neg_datas.append({
                            'token_ids': text_tokened.ids,
                            'segment_ids': text_tokened.type_ids,
                            'entity_heads': entity_heads,
                            'entity_tails': entity_tails,
                            'rels': rels,
                            'sample_subj_head': subj_head_idx,
                            'sample_subj_tail': subj_tail_idx,
                            'sample_rel': neg_rel_idx,
                            'sample_obj_heads': np.zeros(self.max_len),  # set 0 for negative samples
                        })
                        neg_history.add(neg_pair)

                # 3. (neg sub, pos rel)
                for (neg_subj_head_idx, neg_sub_tail_idx) in subj_set - {(subj_head_idx, subj_tail_idx)}:
                    neg_pair = (neg_subj_head_idx, neg_sub_tail_idx, rel_idx)
                    if neg_pair not in subj_rel_set and neg_pair not in neg_history:
                        current_neg_datas.append({
                            'token_ids': text_tokened.ids,
                            'segment_ids': text_tokened.type_ids,
                            'entity_heads': entity_heads,
                            'entity_tails': entity_tails,
                            'rels': rels,
                            'sample_subj_head': neg_subj_head_idx,
                            'sample_subj_tail': neg_sub_tail_idx,
                            'sample_rel': rel_idx,
                            'sample_obj_heads': np.zeros(self.max_len),  # set 0 for negative samples
                        })
                        neg_history.add(neg_pair)

                np.random.shuffle(current_neg_datas)
                if self.neg_samples is not None:
                    current_neg_datas = current_neg_datas[:self.neg_samples]
                neg_datas += current_neg_datas
            current_datas = pos_datas + neg_datas
            self.datas.extend(current_datas)

        self.steps = len(self.datas) // self.batch_size
        if len(self.datas) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random: bool = False):
        idxs = list(range(len(self.datas)))
        if random:
            np.random.shuffle(idxs)
        batch_tokens, batch_segments = [], []
        batch_entity_heads, batch_entity_tails = [], []
        batch_rels = []
        batch_sample_subj_head, batch_sample_subj_tail = [], []
        batch_sample_rel = []
        batch_sample_obj_heads = []

        for idx in idxs:
            obj = self.datas[idx]
            batch_tokens.append(obj['token_ids'])
            batch_segments.append(obj['segment_ids'])
            batch_entity_heads.append(obj['entity_heads'])
            batch_entity_tails.append(obj['entity_tails'])
            batch_rels.append(obj['rels'])
            batch_sample_subj_head.append(obj['sample_subj_head'])
            batch_sample_subj_tail.append(obj['sample_subj_tail'])
            batch_sample_rel.append(obj['sample_rel'])
            batch_sample_obj_heads.append(obj['sample_obj_heads'])
            if len(batch_tokens) == self.batch_size or idx == idxs[-1]:
                batch_tokens = pad_sequences(batch_tokens, maxlen=self.max_len, padding='post', truncating='post')
                batch_segments = pad_sequences(batch_segments, maxlen=self.max_len, padding='post', truncating='post')
                batch_entity_heads = pad_sequences(batch_entity_heads, maxlen=self.max_len, value=np.zeros(2))
                batch_entity_tails = pad_sequences(batch_entity_tails, maxlen=self.max_len, value=np.zeros(2))
                batch_rels = np.array(batch_rels)
                batch_sample_subj_head = np.array(batch_sample_subj_head)
                batch_sample_subj_tail = np.array(batch_sample_subj_tail)
                batch_sample_rel = np.array(batch_sample_rel)
                batch_sample_obj_heads = np.array(batch_sample_obj_heads)
                yield [batch_tokens, batch_segments, batch_entity_heads, batch_entity_tails, batch_rels, batch_sample_subj_head, batch_sample_subj_tail, batch_sample_rel, batch_sample_obj_heads], None
                batch_tokens, batch_segments = [], []
                batch_entity_heads, batch_entity_tails = [], []
                batch_rels = []
                batch_sample_subj_head, batch_sample_subj_tail = [], []
                batch_sample_rel = []
                batch_sample_obj_heads = []

    def forfit(self, random: bool = False):
        while True:
            for inputs, labels in self.__iter__(random=random):
                yield inputs, labels
