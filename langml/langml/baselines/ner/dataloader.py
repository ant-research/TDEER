# -*- coding: utf-8 -*-

import math
from random import shuffle
from typing import Dict, List, Optional

from boltons.iterutils import chunked_iter
import tensorflow as tf
from langml import TF_KERAS
if TF_KERAS:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
else:
    from keras.preprocessing.sequence import pad_sequences


def load_data(fpath: str, build_vocab: bool = False) -> List:
    if build_vocab:
        label2id = {'O': 0}
    datas = []
    with open(fpath, 'r', encoding='utf-8') as reader:
        for sentence in reader.read().split('\n\n'):
            if not sentence:
                continue
            data = []
            for chunk in sentence.split('\n'):
                try:
                    segment, label = chunk.split('\t')
                    if build_vocab:
                        if label != 'O' and f'B-{label}' not in label2id:
                            label2id[f'B-{label}'] = len(label2id)
                        if label != 'O' and f'I-{label}' not in label2id:
                            label2id[f'I-{label}'] = len(label2id)
                    data.append((segment, label))
                except ValueError:
                    print('broken data:', chunk)
            datas.append(data)
    if build_vocab:
        return datas, label2id
    return datas


class DataGenerator:
    def __init__(self,
                 datas: List,
                 tokenizer: object,
                 label2id: Dict,
                 batch_size: int = 32,
                 max_len: Optional[int] = None,
                 is_bert: bool = True):
        self.label2id = label2id
        self.batch_size = batch_size
        self.max_len = max_len
        self.is_bert = is_bert
        start_token_id = tokenizer.token_to_id(tokenizer.special_tokens.CLS)
        end_token_id = tokenizer.token_to_id(tokenizer.special_tokens.SEP)

        self.datas = []
        for data in datas:
            token_ids, labels = [start_token_id], [label2id['O']]
            for segment, label in data:
                tokened = tokenizer.encode(segment)
                token_id = tokened.ids[1:-1]
                token_ids += token_id
                if label == 'O':
                    labels += [label2id['O']] * len(token_id)
                else:
                    labels += ([label2id[f'B-{label}']] + [label2id[f'I-{label}']] * (len(token_id) - 1))
            assert len(token_ids) == len(labels)
            if max_len is not None:
                token_ids = token_ids[:max_len - 1]
                labels = labels[:max_len - 1]
            token_ids += [end_token_id]
            labels += [label2id['O']]
            segment_ids = [0] * len(token_ids)
            self.datas.append((token_ids, segment_ids, labels))

    def __len__(self) -> int:
        return math.ceil(len(self.datas) / self.batch_size)

    def __iter__(self, random: bool = False):
        if random:
            shuffle(self.datas)

        for chunks in chunked_iter(self.datas, self.batch_size):
            batch_tokens, batch_segments, batch_labels = [], [], []

            for token_ids, segment_ids, labels in chunks:
                batch_tokens.append(token_ids)
                batch_segments.append(segment_ids)
                batch_labels.append(labels)

            batch_tokens = pad_sequences(batch_tokens, padding='post', truncating='post')
            batch_segments = pad_sequences(batch_segments, padding='post', truncating='post')
            batch_labels = pad_sequences(batch_labels, padding='post', truncating='post')
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
                 datas: List,
                 tokenizer: object,
                 label2id: Dict,
                 batch_size: int = 32,
                 max_len: Optional[int] = None,
                 is_bert: bool = True):
        super().__init__(datas, tokenizer, label2id, batch_size=batch_size, max_len=max_len, is_bert=is_bert)

    def __call__(self):
        def gen_features():
            for token_ids, segment_ids, label_ids in self.datas:
                token_ids = token_ids[:self.max_len] + [0] * (self.max_len - len(token_ids))
                label_ids = label_ids[:self.max_len] + [self.label2id['O']] * (self.max_len - len(label_ids))
                if self.is_bert:
                    segment_ids = [0] * len(token_ids)
                    yield {'Input-Token': token_ids, 'Input-Segment': segment_ids}, label_ids
                else:
                    yield token_ids, label_ids

        if self.is_bert:
            output_types = ({'Input-Token': tf.int64, 'Input-Segment': tf.int64}, tf.int64)
            output_shapes = ({'Input-Token': tf.TensorShape((None, )),
                              'Input-Segment': tf.TensorShape((None, ))},
                             tf.TensorShape((None, )))
        else:
            output_types = (tf.int64, tf.int64)
            output_shapes = (tf.TensorShape((None, )), tf.TensorShape((None, )))
        dataset = tf.data.Dataset.from_generator(gen_features,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self.batch_size * 1000)
        dataset = dataset.batch(self.batch_size)
        return dataset
