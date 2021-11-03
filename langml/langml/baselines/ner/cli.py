# -*- coding: utf-8 -*-

import os
import json
from typing import Optional
from shutil import copyfile

from langml import TF_KERAS, TF_VERSION
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
else:
    import keras
    import keras.backend as K
import click

from langml.log import info
from langml.tokenizer import WPTokenizer, SPTokenizer
from langml.baselines import Parameters
from langml.baselines.ner import report_detail_metrics
from langml.baselines.ner.dataloader import load_data, DataGenerator, TFDataGenerator
from langml.model import save_frozen


@click.group()
def ner():
    """ner command line tools"""
    pass


@ner.command()
@click.option('--backbone', type=str, default='bert',
              help='specify backbone: bert | roberta | albert')
@click.option('--epoch', type=int, default=20, help='epochs')
@click.option('--batch_size', type=int, default=32, help='batch size')
@click.option('--learning_rate', type=float, default=2e-5, help='learning rate')
@click.option('--dropout_rate', type=float, default=0.2, help='dropout rate')
@click.option('--max_len', type=int, default=512, help='max len')
@click.option('--lowercase', is_flag=True, default=False, help='do lowercase')
@click.option('--tokenizer_type', type=str, default=None,
              help='specify tokenizer type from [`wordpiece`, `sentencepiece`]')
@click.option('--config_path', type=str, required=True, help='bert config path')
@click.option('--ckpt_path', type=str, required=True, help='bert checkpoint path')
@click.option('--vocab_path', type=str, required=True, help='bert vocabulary path')
@click.option('--train_path', type=str, required=True, help='train path')
@click.option('--dev_path', type=str, required=True, help='dev path')
@click.option('--test_path', type=str, default=None, help='test path')
@click.option('--save_dir', type=str, required=True, help='dir to save model')
@click.option('--early_stop', type=int, default=10, help='patience to early stop')
@click.option('--distribute', is_flag=True, default=False, help='distributed training')
@click.option('--verbose', type=int, default=2, help='0 = silent, 1 = progress bar, 2 = one line per epoch')
def bert_crf(backbone: str, epoch: int, batch_size: int, learning_rate: float,
             dropout_rate: float, max_len: Optional[int], lowercase: bool,
             tokenizer_type: Optional[str], config_path: str, ckpt_path: str,
             vocab_path: str, train_path: str, dev_path: str, test_path: str,
             save_dir: str, early_stop: int, distribute: bool, verbose: int):

    from langml.baselines.ner.bert_crf import BertCRF

    # check distribute
    if distribute:
        assert TF_KERAS, 'Please `export TF_KERAS=1` to support distributed training!'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_datas, label2id = load_data(train_path, build_vocab=True)
    id2label = {v: k for k, v in label2id.items()}
    dev_datas = load_data(dev_path)
    test_datas = None
    if test_path is not None:
        test_datas = load_data(test_path)
    info(f'labels: {label2id}')
    info(f'train size: {len(train_datas)}')
    info(f'valid size: {len(dev_datas)}')
    if test_path is not None:
        info(f'test size: {len(test_datas)}')

    if tokenizer_type == 'wordpiece':
        tokenizer = WPTokenizer(vocab_path, lowercase=lowercase)
    elif tokenizer_type == 'sentencepiece':
        tokenizer = SPTokenizer(vocab_path, lowercase=lowercase)
    else:
        # auto deduce
        if vocab_path.endswith('.txt'):
            info('automatically apply `WPTokenizer`')
            tokenizer = WPTokenizer(vocab_path, lowercase=lowercase)
        elif vocab_path.endswith('.model'):
            info('automatically apply `SPTokenizer`')
            tokenizer = SPTokenizer(vocab_path, lowercase=lowercase)
        else:
            raise ValueError("Langml cannot deduce which tokenizer to apply, please specify `tokenizer_type` manually.")  # NOQA

    tokenizer.enable_truncation(max_length=max_len)
    params = Parameters({
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'tag_size': len(label2id),
        'vocab_size': tokenizer.get_vocab_size(),
    })
    if distribute:
        import tensorflow as tf
        # distributed training
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = BertCRF(config_path, ckpt_path, params, backbone=backbone).build_model(lazy_restore=True)
    else:
        model = BertCRF(config_path, ckpt_path, params, backbone=backbone).build_model()

    if distribute:
        info('distributed training! using `TFDataGenerator`')
        assert max_len is not None, 'Please specify `max_len`!'
        train_generator = TFDataGenerator(train_datas, tokenizer, label2id,
                                          batch_size=batch_size, max_len=max_len, is_bert=True)
        dev_generator = TFDataGenerator(dev_datas, tokenizer, label2id,
                                        batch_size=batch_size, max_len=max_len, is_bert=True)
        train_dataset = train_generator()
        dev_dataset = dev_generator()
    else:
        train_generator = DataGenerator(train_datas, tokenizer, label2id,
                                        batch_size=batch_size, max_len=max_len, is_bert=True)
        dev_generator = DataGenerator(dev_datas, tokenizer, label2id,
                                      batch_size=batch_size, max_len=max_len, is_bert=True)
        train_dataset = train_generator.forfit(random=True)
        dev_dataset = dev_generator.forfit(random=False)

    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor='val_viterbi_accuracy',
        min_delta=1e-4,
        patience=early_stop,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    save_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(save_dir, 'best_model.weights'),
        save_best_only=True,
        save_weights_only=True,
        monitor='val_viterbi_accuracy',
        mode='auto')
    model.fit(train_dataset,
              steps_per_epoch=len(train_generator),
              validation_data=dev_dataset,
              validation_steps=len(dev_generator),
              verbose=verbose,
              epochs=epoch,
              callbacks=[early_stop_callback, save_checkpoint_callback])
    # clear model
    del model
    if distribute:
        del strategy
    K.clear_session()
    # restore model
    model = BertCRF(config_path, ckpt_path, params, backbone=backbone).build_model()
    if TF_KERAS or TF_VERSION > 1:
        model.load_weights(os.path.join(save_dir, 'best_model.weights')).expect_partial()
    else:
        model.load_weights(os.path.join(save_dir, 'best_model.weights'))
    # compute detail metrics
    info('done to training! start to compute detail metrics...')
    print('develop metrics:')
    dev_generator = DataGenerator(dev_datas, tokenizer, label2id,
                                  batch_size=batch_size, max_len=max_len, is_bert=True)
    report_detail_metrics(model, dev_generator.datas, id2label, is_bert=True)
    if test_datas:
        print('test metrics:')
        test_generator = DataGenerator(test_datas, tokenizer, label2id,
                                       batch_size=batch_size, max_len=max_len, is_bert=True)
        report_detail_metrics(model, test_generator.datas, id2label, is_bert=True)
    # save model
    info('start to save frozen')
    save_frozen(model, os.path.join(save_dir, 'frozen_model'))
    info('start to save label')
    with open(os.path.join(save_dir, 'label2id.json'), 'w', encoding='utf-8') as writer:
        json.dump(label2id, writer)
    info('copy vocab')
    copyfile(vocab_path, os.path.join(save_dir, 'vocab.txt'))


@ner.command()
@click.option('--epoch', type=int, default=20, help='epochs')
@click.option('--batch_size', type=int, default=32, help='batch size')
@click.option('--learning_rate', type=float, default=1e-3, help='learning rate')
@click.option('--dropout_rate', type=float, default=0.2, help='dropout rate')
@click.option('--embedding_size', type=int, default=200, help='embedding size')
@click.option('--hidden_size', type=int, default=128, help='hidden size')
@click.option('--max_len', type=int, default=None, help='max len')
@click.option('--lowercase', is_flag=True, default=False, help='do lowercase')
@click.option('--tokenizer_type', type=str, default=None,
              help='specify tokenizer type from [`wordpiece`, `sentencepiece`]')
@click.option('--vocab_path', type=str, required=True, help='vocabulary path')
@click.option('--train_path', type=str, required=True, help='train path')
@click.option('--dev_path', type=str, required=True, help='dev path')
@click.option('--test_path', type=str, default=None, help='test path')
@click.option('--save_dir', type=str, required=True, help='dir to save model')
@click.option('--early_stop', type=int, default=10, help='patience to early stop')
@click.option('--distribute', is_flag=True, default=False, help='distributed training')
@click.option('--verbose', type=int, default=2, help='0 = silent, 1 = progress bar, 2 = one line per epoch')
def lstm_crf(epoch: int, batch_size: int, learning_rate: float, dropout_rate: float,
             embedding_size: int, hidden_size: int, max_len: Optional[int],
             lowercase: bool, tokenizer_type: Optional[str], vocab_path: str,
             train_path: str, dev_path: str, test_path: str, save_dir: str,
             early_stop: int, distribute: bool, verbose: int):

    from langml.baselines.ner.lstm_crf import LstmCRF

    # check distribute
    if distribute:
        assert TF_KERAS, 'Please `export TF_KERAS=1` to support distributed training!'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_datas, label2id = load_data(train_path, build_vocab=True)
    id2label = {v: k for k, v in label2id.items()}
    dev_datas = load_data(dev_path)
    test_datas = None
    if test_path is not None:
        test_datas = load_data(test_path)
    info(f'labels: {label2id}')
    info(f'train size: {len(train_datas)}')
    info(f'valid size: {len(dev_datas)}')
    if test_path is not None:
        info(f'test size: {len(test_datas)}')

    if tokenizer_type == 'wordpiece':
        tokenizer = WPTokenizer(vocab_path, lowercase=lowercase)
    elif tokenizer_type == 'sentencepiece':
        tokenizer = SPTokenizer(vocab_path, lowercase=lowercase)
    else:
        # auto deduce
        if vocab_path.endswith('.txt'):
            info('automatically apply `WPTokenizer`')
            tokenizer = WPTokenizer(vocab_path, lowercase=lowercase)
        elif vocab_path.endswith('.model'):
            info('automatically apply `SPTokenizer`')
            tokenizer = SPTokenizer(vocab_path, lowercase=lowercase)
        else:
            raise ValueError("Langml cannot deduce which tokenizer to apply, please specify `tokenizer_type` manually.")  # NOQA

    if max_len is not None:
        tokenizer.enable_truncation(max_length=max_len)
    params = Parameters({
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'tag_size': len(label2id),
        'vocab_size': tokenizer.get_vocab_size(),
        'embedding_size': embedding_size,
        'hidden_size': hidden_size
    })
    if distribute:
        import tensorflow as tf
        # distributed training
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = LstmCRF(params).build_model()
    else:
        model = LstmCRF(params).build_model()

    if distribute:
        info('distributed training! using `TFDataGenerator`')
        assert max_len is not None, 'Please specify `max_len`!'
        train_generator = TFDataGenerator(train_datas, tokenizer, label2id,
                                          batch_size=batch_size, max_len=max_len, is_bert=False)
        dev_generator = TFDataGenerator(dev_datas, tokenizer, label2id,
                                        batch_size=batch_size, max_len=max_len, is_bert=False)
        train_dataset = train_generator()
        dev_dataset = dev_generator()
    else:
        train_generator = DataGenerator(train_datas, tokenizer, label2id,
                                        batch_size=batch_size, max_len=max_len, is_bert=False)
        dev_generator = DataGenerator(dev_datas, tokenizer, label2id,
                                      batch_size=batch_size, max_len=max_len, is_bert=False)
        train_dataset = train_generator.forfit(random=True)
        dev_dataset = dev_generator.forfit(random=False)

    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor='val_viterbi_accuracy',
        min_delta=1e-4,
        patience=early_stop,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    save_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(save_dir, 'best_model.weights'),
        save_best_only=True,
        save_weights_only=True,
        monitor='val_viterbi_accuracy',
        mode='auto')
    model.fit(train_dataset,
              steps_per_epoch=len(train_generator),
              validation_data=dev_dataset,
              validation_steps=len(dev_generator),
              verbose=verbose,
              epochs=epoch,
              callbacks=[early_stop_callback, save_checkpoint_callback])
    # clear model
    del model
    if distribute:
        del strategy
    K.clear_session()
    # restore model
    model = LstmCRF(params).build_model()
    if TF_KERAS or TF_VERSION > 1:
        model.load_weights(os.path.join(save_dir, 'best_model.weights')).expect_partial()
    else:
        model.load_weights(os.path.join(save_dir, 'best_model.weights'))
    # compute detail metrics
    info('done to training! start to compute detail metrics...')
    print('develop metrics:')
    dev_generator = DataGenerator(dev_datas, tokenizer, label2id,
                                  batch_size=batch_size, max_len=max_len, is_bert=False)
    report_detail_metrics(model, dev_generator.datas, id2label, is_bert=False)
    if test_datas:
        print('test metrics:')
        test_generator = DataGenerator(test_datas, tokenizer, label2id,
                                       batch_size=batch_size, max_len=max_len, is_bert=False)
        report_detail_metrics(model, test_generator.datas, id2label, is_bert=False)
    # save model
    info('start to save frozen')
    save_frozen(model, os.path.join(save_dir, 'frozen_model'))
    info('start to save label')
    with open(os.path.join(save_dir, 'label2id.json'), 'w', encoding='utf-8') as writer:
        json.dump(label2id, writer)
    info('copy vocab')
    copyfile(vocab_path, os.path.join(save_dir, 'vocab.txt'))
