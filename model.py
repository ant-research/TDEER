# -*- coding:utf-8 -*-

import os
from typing import List, Tuple

import tensorflow as tf
import keras.layers as L
import keras.backend as K
from keras.models import Model
from keras.callbacks import Callback
from keras.optimizers import Adam
from langml.plm.bert import load_bert
from langml.layers import SelfAttention
from langml.tensor_typing import Models

from utils import compute_metrics


def build_model(bert_dir: str, learning_rate: float, relation_size: int) -> Tuple[Models, Models, Models, Models]:

    def gather_span(x):
        seq, idxs = x
        idxs = K.cast(idxs, 'int32')
        if len(K.int_shape(idxs)) == 3:
            res = []
            for i in range(len(K.int_shape(idxs))):
                batch_idxs = K.arange(0, K.shape(seq)[0])
                batch_idxs = K.expand_dims(batch_idxs, 1)
                indices = K.concatenate([batch_idxs, idxs[:, i, :]], 1)
                res.append(K.expand_dims(tf.gather_nd(seq, indices), 1))
            return K.concatenate(res, 1)
        batch_idxs = K.arange(0, K.shape(seq)[0])
        batch_idxs = K.expand_dims(batch_idxs, 1)
        idxs = K.concatenate([batch_idxs, idxs], 1)
        return tf.gather_nd(seq, idxs)

    bert_model, _ = load_bert(
        config_path=os.path.join(bert_dir, 'bert_config.json'),
        checkpoint_path=os.path.join(bert_dir, 'bert_model.ckpt'),
    )

    # entities in
    gold_entity_heads_in = L.Input(shape=(None, 2), name='gold_entity_heads')
    gold_entity_tails_in = L.Input(shape=(None, 2), name='gold_entity_tails')
    gold_rels_in = L.Input(shape=(relation_size, ), name='gold_rels')
    # pos sample
    sub_head_in = L.Input(shape=(1,), name='sample_subj_head')
    sub_tail_in = L.Input(shape=(1,), name='sample_subj_tail')
    rel_in = L.Input(shape=(1,), name='sample_rel')
    gold_obj_head_in = L.Input(shape=(None, ), name='sample_obj_heads')

    gold_entity_heads, gold_entity_tails, sub_head, sub_tail, rel, gold_rels, gold_obj_head = gold_entity_heads_in, gold_entity_tails_in, sub_head_in, sub_tail_in, rel_in, gold_rels_in, gold_obj_head_in
    mask = L.Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(bert_model.input[0])

    tokens_feature = bert_model.output

    # predict relations
    pred_rels = L.Lambda(lambda x: x[:, 0])(tokens_feature)
    pred_rels = L.Dense(relation_size, activation='sigmoid', name='pred_rel')(pred_rels)
    rel_model = Model(bert_model.input, [pred_rels])

    # predict entity
    pred_entity_heads = L.Dense(2, activation='sigmoid', name='entity_heads')(tokens_feature)
    pred_entity_tails = L.Dense(2, activation='sigmoid', name='entity_tails')(tokens_feature)
    entity_model = Model(bert_model.input, [pred_entity_heads, pred_entity_tails])

    # predict object
    tokens_feature_size = K.int_shape(tokens_feature)[-1]
    sub_head_feature = L.Lambda(gather_span)([tokens_feature, sub_head])
    sub_head_feature = L.Lambda(lambda x: K.expand_dims(x, 1))(sub_head_feature)
    sub_tail_feature = L.Lambda(gather_span)([tokens_feature, sub_tail])
    sub_tail_feature = L.Lambda(lambda x: K.expand_dims(x, 1))(sub_tail_feature)
    sub_feature = L.Average()([sub_head_feature, sub_tail_feature])

    rel_feature = L.Embedding(relation_size, tokens_feature_size)(rel)
    rel_feature = L.Dense(tokens_feature_size, activation='relu')(rel_feature)

    obj_feature = L.Add()([tokens_feature, sub_feature, rel_feature])

    value = SelfAttention(is_residual=True, attention_activation='relu')(obj_feature)

    pred_obj_head = L.Dense(1, activation='sigmoid', name='pred_obj_head')(value)

    translate_model = Model((*bert_model.input, sub_head_in, sub_tail_in, rel_in), [pred_obj_head])

    train_model = Model(inputs=(*bert_model.input, gold_entity_heads_in, gold_entity_tails_in, gold_rels_in, sub_head_in, sub_tail_in, rel_in, gold_obj_head_in),
                        outputs=[pred_entity_heads, pred_entity_tails, pred_rels, pred_obj_head])

    # entity loss
    entity_heads_loss = K.sum(K.binary_crossentropy(gold_entity_heads, pred_entity_heads), 2, keepdims=True)
    entity_heads_loss = K.sum(entity_heads_loss * mask) / K.sum(mask)
    entity_tails_loss = K.sum(K.binary_crossentropy(gold_entity_tails, pred_entity_tails), 2, keepdims=True)
    entity_tails_loss = K.sum(entity_tails_loss * mask) / K.sum(mask)

    # rel loss
    rel_loss = K.mean(K.binary_crossentropy(gold_rels, pred_rels))

    # obj loss
    gold_obj_head = K.expand_dims(gold_obj_head, 2)
    obj_head_loss = K.binary_crossentropy(gold_obj_head, pred_obj_head)
    obj_head_loss = K.sum(obj_head_loss * mask) / K.sum(mask)

    # joint loss
    loss = (entity_heads_loss + entity_tails_loss) + rel_loss + 5.0 * obj_head_loss

    train_model.add_loss(loss)
    train_model.compile(optimizer=Adam(learning_rate))
    train_model.summary()

    return entity_model, rel_model, translate_model, train_model


class Evaluator(Callback):
    def __init__(self,
                 infer: object,
                 train_model: Models,
                 dev_data: List,
                 save_weights_path: str,
                 model_name: str,
                 learning_rate: float = 5e-5,
                 min_learning_rate: float = 5e-6):
        self.infer = infer
        self.train_model = train_model
        self.dev_data = dev_data
        self.save_weights_path = save_weights_path
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.passed = 0
        self.is_exact_match = True if self.model_name.startswith('NYT11-HRL') else False

    def on_train_begin(self, logs=None):
        self.best = float('-inf')

    def on_batch_begin(self, batch, logs=None):
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * self.learning_rate
            K.set_value(self.train_model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (self.learning_rate - self.min_learning_rate)
            lr += self.min_learning_rate
            K.set_value(self.train_model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        precision, recall, f1 = compute_metrics(self.infer, self.dev_data, exact_match=self.is_exact_match, model_name=self.model_name)
        if f1 > self.best:
            self.best = f1
            self.train_model.save_weights(self.save_weights_path)
            print("new best result!")
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
