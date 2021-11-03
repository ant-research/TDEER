# -*- coding: utf-8 -*-

import json
import math
from random import shuffle
from typing import Dict, List

import numpy as np
from boltons.iterutils import chunked_iter
import tensorflow as tf
from langml import TF_KERAS
if TF_KERAS:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
else:
    from keras.preprocessing.sequence import pad_sequences


def load_data(fpath: str, build_vocab: bool = False) -> List:
    if build_vocab:
        label2id = {}
    datas = []
    with open(fpath, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if build_vocab and obj['label'] not in label2id:
                label2id[obj['label']] = len(label2id)
            datas.append((obj['text'], obj['label']))
    if build_vocab:
        return datas, label2id
    return datas


class DataGenerator:
    def __init__(self,
                 datas: List,
                 tokenizer: object,
                 label2id: Dict,
                 batch_size: int = 32,
                 is_bert: bool = True):
        self.batch_size = batch_size
        self.is_bert = is_bert

        self.datas = []
        for text, label in datas:
            tokened = tokenizer.encode(text)
            self.datas.append((tokened.ids, tokened.segment_ids, label2id[label]))

    def __len__(self) -> int:
        return math.ceil(len(self.datas) / self.batch_size)

    def __iter__(self, random: bool = False):
        if random:
            shuffle(self.datas)

        for chunks in chunked_iter(self.datas, self.batch_size):
            batch_tokens, batch_segments, batch_labels = [], [], []

            for token_ids, segment_ids, label in chunks:
                batch_tokens.append(token_ids)
                batch_segments.append(segment_ids)
                batch_labels.append([label])

            batch_tokens = pad_sequences(batch_tokens, padding='post', truncating='post')
            batch_segments = pad_sequences(batch_segments, padding='post', truncating='post')
            batch_labels = np.array(batch_labels)
            if self.is_bert:
                yield [batch_tokens, batch_segments], batch_labels
            else:
                yield batch_tokens, batch_labels

    def forfit(self, random: bool = False):
        while True:
            for inputs, labels in self.__iter__(random=random):
                yield inputs, labels


class TFDataGenerator(DataGenerator):
    def __init__(self,
                 max_len: int,
                 datas: List,
                 tokenizer: object,
                 label2id: Dict,
                 batch_size: int,
                 is_bert: bool):
        super().__init__(datas, tokenizer, label2id, batch_size=batch_size, is_bert=is_bert)
        self.max_len = max_len

    def __call__(self):
        def gen_features():
            for token_ids, _, label in self.datas:
                token_ids = token_ids[: self.max_len] + [0]*(self.max_len - len(token_ids))
                if self.is_bert:
                    segment_ids = [0] * len(token_ids)
                    yield {'Input-Token': token_ids, 'Input-Segment': segment_ids}, [label]
                else:
                    yield token_ids, [label]

        if self.is_bert:
            output_types = ({'Input-Token': tf.int64, 'Input-Segment': tf.int64}, tf.int64)
            output_shapes = ({'Input-Token': tf.TensorShape((None, )),
                              'Input-Segment': tf.TensorShape((None, ))},
                             tf.TensorShape((1, )))
        else:
            output_types = (tf.int64, tf.int64)
            output_shapes = (tf.TensorShape((None, )), tf.TensorShape((1, )))
        dataset = tf.data.Dataset.from_generator(gen_features,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self.batch_size * 1000)
        dataset = dataset.batch(self.batch_size)
        return dataset
