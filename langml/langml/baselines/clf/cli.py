# -*- coding: utf-8 -*-

import os
import json
from typing import Optional
from shutil import copyfile

import click
from langml import TF_VERSION, TF_KERAS
if TF_KERAS:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
else:
    import keras
    import keras.backend as K

from langml.log import info
from langml.baselines import Parameters
from langml.baselines.clf import Infer, compute_detail_metrics
from langml.baselines.clf.dataloader import load_data, DataGenerator, TFDataGenerator
from langml.model import save_frozen
from langml.tokenizer import WPTokenizer, SPTokenizer


MONITOR = 'val_accuracy' if not TF_KERAS or TF_VERSION > 1 else 'val_acc'


@click.group()
def clf():
    """classification command line tools"""
    pass


@clf.command()
@click.option('--backbone', type=str, default='bert',
              help='specify backbone: bert | roberta | albert')
@click.option('--epoch', type=int, default=20, help='epochs')
@click.option('--batch_size', type=int, default=32, help='batch size')
@click.option('--learning_rate', type=float, default=2e-5, help='learning rate')
@click.option('--max_len', type=int, default=512, help='max len')
@click.option('--lowercase', is_flag=True, default=False, help='do lowercase')
@click.option('--tokenizer_type', type=str, default=None,
              help='specify tokenizer type from [`wordpiece`, `sentencepiece`]')
@click.option('--early_stop', type=int, default=10, help='patience to early stop')
@click.option('--use_micro', is_flag=True, default=False, help='whether to use micro metrics')
@click.option('--config_path', type=str, required=True, help='bert config path')
@click.option('--ckpt_path', type=str, required=True, help='bert checkpoint path')
@click.option('--vocab_path', type=str, required=True, help='bert vocabulary path')
@click.option('--train_path', type=str, required=True, help='train path')
@click.option('--dev_path', type=str, required=True, help='dev path')
@click.option('--test_path', type=str, default=None, help='test path')
@click.option('--save_dir', type=str, required=True, help='dir to save model')
@click.option('--verbose', type=int, default=2, help='0 = silent, 1 = progress bar, 2 = one line per epoch')
@click.option('--distribute', is_flag=True, default=False, help='distributed training')
def bert(backbone: str, epoch: int, batch_size: int, learning_rate: float, max_len: Optional[int],
         lowercase: bool, tokenizer_type: Optional[str], early_stop: int, use_micro: bool,
         config_path: str, ckpt_path: str, vocab_path: str, train_path: str, dev_path: str,
         test_path: str, save_dir: str, verbose: int, distribute: bool):

    # check distribute
    if distribute:
        assert TF_KERAS, 'Please `export TF_KERAS=1` to support distributed training!'

    from langml.baselines.clf.bert import Bert

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
        'tag_size': len(label2id),
    })
    if distribute:
        import tensorflow as tf
        # distributed training
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = Bert(config_path, ckpt_path, params, backbone=backbone).build_model(lazy_restore=True)
    else:
        model = Bert(config_path, ckpt_path, params, backbone=backbone).build_model()

    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor=MONITOR,
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
        monitor=MONITOR,
        mode='auto')

    if distribute:
        info('distributed training! using `TFDataGenerator`')
        assert max_len is not None, 'Please specify `max_len`!'
        train_generator = TFDataGenerator(max_len, train_datas, tokenizer, label2id,
                                          batch_size=batch_size, is_bert=True)
        dev_generator = TFDataGenerator(max_len, dev_datas, tokenizer, label2id,
                                        batch_size=batch_size, is_bert=True)
        train_dataset = train_generator()
        dev_dataset = dev_generator()
    else:
        train_generator = DataGenerator(train_datas, tokenizer, label2id,
                                        batch_size=batch_size, is_bert=True)
        dev_generator = DataGenerator(dev_datas, tokenizer, label2id,
                                      batch_size=batch_size, is_bert=True)
        train_dataset = train_generator.forfit(random=True)
        dev_dataset = dev_generator.forfit(random=False)

    model.fit(train_dataset,
              steps_per_epoch=len(train_generator),
              verbose=verbose,
              epochs=epoch,
              validation_data=dev_dataset,
              validation_steps=len(dev_generator),
              callbacks=[early_stop_callback, save_checkpoint_callback])

    # clear model
    del model
    if distribute:
        del strategy
    K.clear_session()
    # restore model
    model = Bert(config_path, ckpt_path, params, backbone=backbone).build_model()
    if TF_KERAS or TF_VERSION > 1:
        model.load_weights(os.path.join(save_dir, 'best_model.weights')).expect_partial()
    else:
        model.load_weights(os.path.join(save_dir, 'best_model.weights'))
    # compute detail metrics
    info('done to training! start to compute detail metrics...')
    infer = Infer(model, tokenizer, id2label, is_bert=True)
    _, _, dev_cr = compute_detail_metrics(infer, dev_datas, use_micro=use_micro)
    print('develop metrics:')
    print(dev_cr)
    if test_datas:
        _, _, test_cr = compute_detail_metrics(infer, test_datas, use_micro=use_micro)
        print('test metrics:')
        print(test_cr)
    # save model
    info('start to save frozen')
    save_frozen(model, os.path.join(save_dir, 'frozen_model'))
    info('start to save label')
    with open(os.path.join(save_dir, 'label2id.json'), 'w', encoding='utf-8') as writer:
        json.dump(label2id, writer)
    info('copy vocab')
    copyfile(vocab_path, os.path.join(save_dir, 'vocab.txt'))


@clf.command()
@click.option('--epoch', type=int, default=20, help='epochs')
@click.option('--batch_size', type=int, default=32, help='batch size')
@click.option('--learning_rate', type=float, default=1e-3, help='learning rate')
@click.option('--embedding_size', type=int, default=200, help='embedding size')
@click.option('--filter_size', type=int, default=100, help='filter size of convolution')
@click.option('--max_len', type=int, default=None, help='max len')
@click.option('--lowercase', is_flag=True, default=False, help='do lowercase')
@click.option('--tokenizer_type', type=str, default=None,
              help='specify tokenizer type from [`wordpiece`, `sentencepiece`]')
@click.option('--early_stop', type=int, default=10, help='patience to early stop')
@click.option('--use_micro', is_flag=True, default=False, help='whether to use micro metrics')
@click.option('--vocab_path', type=str, required=True, help='vocabulary path')
@click.option('--train_path', type=str, required=True, help='train path')
@click.option('--dev_path', type=str, required=True, help='dev path')
@click.option('--test_path', type=str, default=None, help='test path')
@click.option('--save_dir', type=str, required=True, help='dir to save model')
@click.option('--verbose', type=int, default=2, help='0 = silent, 1 = progress bar, 2 = one line per epoch')
@click.option('--distribute', is_flag=True, default=False, help='distributed training')
def textcnn(epoch: int, batch_size: int, learning_rate: float, embedding_size: int,
            filter_size: int, max_len: Optional[int], lowercase: bool, tokenizer_type: Optional[str],
            early_stop: int, use_micro: bool, vocab_path: str, train_path: str, dev_path: str,
            test_path: str, save_dir: str, verbose: int, distribute: bool):

    # check distribute
    if distribute:
        assert TF_KERAS, 'please `export TF_KERAS=1` to support distributed training!'

    from langml.baselines.clf.textcnn import TextCNN

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
        'tag_size': len(label2id),
        'vocab_size': tokenizer.get_vocab_size(),
        'embedding_size': embedding_size,
        'filter_size': filter_size
    })
    if distribute:
        import tensorflow as tf
        # distributed training
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = TextCNN(params).build_model()
    else:
        model = TextCNN(params).build_model()

    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor=MONITOR,
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
        monitor=MONITOR,
        mode='auto')

    if distribute:
        info('distributed training! using `TFDataGenerator`')
        assert max_len is not None, 'Please specify `max_len`!'
        train_generator = TFDataGenerator(max_len, train_datas, tokenizer, label2id,
                                          batch_size=batch_size, is_bert=False)
        dev_generator = TFDataGenerator(max_len, dev_datas, tokenizer, label2id,
                                        batch_size=batch_size, is_bert=False)
        train_dataset = train_generator()
        dev_dataset = dev_generator()
    else:
        train_generator = DataGenerator(train_datas, tokenizer, label2id,
                                        batch_size=batch_size, is_bert=False)
        dev_generator = DataGenerator(dev_datas, tokenizer, label2id,
                                      batch_size=batch_size, is_bert=False)
        train_dataset = train_generator.forfit(random=True)
        dev_dataset = dev_generator.forfit(random=False)

    model.fit(train_dataset,
              steps_per_epoch=len(train_generator),
              verbose=verbose,
              epochs=epoch,
              validation_data=dev_dataset,
              validation_steps=len(dev_generator),
              callbacks=[early_stop_callback, save_checkpoint_callback])

    # clear model
    del model
    if distribute:
        del strategy
    K.clear_session()
    # restore model
    model = TextCNN(params).build_model()
    if TF_KERAS or TF_VERSION > 1:
        model.load_weights(os.path.join(save_dir, 'best_model.weights')).expect_partial()
    else:
        model.load_weights(os.path.join(save_dir, 'best_model.weights'))
    # compute detail metrics
    info('done to training! start to compute detail metrics...')
    infer = Infer(model, tokenizer, id2label, is_bert=False)
    _, _, dev_cr = compute_detail_metrics(infer, dev_datas, use_micro=use_micro)
    print('develop metrics:')
    print(dev_cr)
    if test_datas:
        _, _, test_cr = compute_detail_metrics(infer, test_datas, use_micro=use_micro)
        print('test metrics:')
        print(test_cr)
    # save model
    info('start to save frozen')
    save_frozen(model, os.path.join(save_dir, 'frozen_model'))
    info('start to save label')
    with open(os.path.join(save_dir, 'label2id.json'), 'w', encoding='utf-8') as writer:
        json.dump(label2id, writer)
    info('copy vocab')
    copyfile(vocab_path, os.path.join(save_dir, 'vocab.txt'))


@clf.command()
@click.option('--epoch', type=int, default=20, help='epochs')
@click.option('--batch_size', type=int, default=32, help='batch size')
@click.option('--learning_rate', type=float, default=1e-3, help='learning rate')
@click.option('--embedding_size', type=int, default=200, help='embedding size')
@click.option('--hidden_size', type=int, default=128, help='hidden size of lstm')
@click.option('--max_len', type=int, default=None, help='max len')
@click.option('--lowercase', is_flag=True, default=False, help='do lowercase')
@click.option('--tokenizer_type', type=str, default=None,
              help='specify tokenizer type from [`wordpiece`, `sentencepiece`]')
@click.option('--early_stop', type=int, default=10, help='patience to early stop')
@click.option('--use_micro', is_flag=True, default=False, help='whether to use micro metrics')
@click.option('--vocab_path', type=str, required=True, help='vocabulary path')
@click.option('--train_path', type=str, required=True, help='train path')
@click.option('--dev_path', type=str, required=True, help='dev path')
@click.option('--test_path', type=str, default=None, help='test path')
@click.option('--save_dir', type=str, required=True, help='dir to save model')
@click.option('--verbose', type=int, default=2, help='0 = silent, 1 = progress bar, 2 = one line per epoch')
@click.option('--with_attention', is_flag=True, default=False, help='apply bilstm attention')
@click.option('--distribute', is_flag=True, default=False, help='distributed training')
def bilstm(epoch: int, batch_size: int, learning_rate: float, embedding_size: int,
           hidden_size: int, max_len: Optional[int], lowercase: bool, tokenizer_type: Optional[str],
           early_stop: int, use_micro: bool, vocab_path: str, train_path: str, dev_path: str,
           test_path: str, save_dir: str, verbose: int, with_attention: bool, distribute: bool):

    # check distribute
    if distribute:
        assert TF_KERAS, 'please `export TF_KERAS=1` to support distributed training!'

    from langml.baselines.clf.bilstm import BiLSTM as BiLSTM

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
            model = BiLSTM(params).build_model(with_attention=with_attention)
    else:
        model = BiLSTM(params).build_model(with_attention=with_attention)

    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor=MONITOR,
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
        monitor=MONITOR,
        mode='auto')

    if distribute:
        info('distributed training! using `TFDataGenerator`')
        assert max_len is not None, 'Please specify `max_len`!'
        train_generator = TFDataGenerator(max_len, train_datas, tokenizer, label2id,
                                          batch_size=batch_size, is_bert=False)
        dev_generator = TFDataGenerator(max_len, dev_datas, tokenizer, label2id,
                                        batch_size=batch_size, is_bert=False)
        train_dataset = train_generator()
        dev_dataset = dev_generator()
    else:
        train_generator = DataGenerator(train_datas, tokenizer, label2id,
                                        batch_size=batch_size, is_bert=False)
        dev_generator = DataGenerator(dev_datas, tokenizer, label2id,
                                      batch_size=batch_size, is_bert=False)
        train_dataset = train_generator.forfit(random=True)
        dev_dataset = dev_generator.forfit(random=False)

    model.fit(train_dataset,
              steps_per_epoch=len(train_generator),
              verbose=verbose,
              epochs=epoch,
              validation_data=dev_dataset,
              validation_steps=len(dev_generator),
              callbacks=[early_stop_callback, save_checkpoint_callback])

    # clear model
    del model
    if distribute:
        del strategy
    K.clear_session()
    # restore model
    model = BiLSTM(params).build_model(with_attention=with_attention)
    if TF_KERAS or TF_VERSION > 1:
        model.load_weights(os.path.join(save_dir, 'best_model.weights')).expect_partial()
    else:
        model.load_weights(os.path.join(save_dir, 'best_model.weights'))
    # compute detail metrics
    info('done to training! start to compute detail metrics...')
    infer = Infer(model, tokenizer, id2label, is_bert=False)
    _, _, dev_cr = compute_detail_metrics(infer, dev_datas, use_micro=use_micro)
    print('develop metrics:')
    print(dev_cr)
    if test_datas:
        _, _, test_cr = compute_detail_metrics(infer, test_datas, use_micro=use_micro)
        print('test metrics:')
        print(test_cr)
    # save model
    info('start to save frozen')
    save_frozen(model, os.path.join(save_dir, 'frozen_model'))
    info('start to save label')
    with open(os.path.join(save_dir, 'label2id.json'), 'w', encoding='utf-8') as writer:
        json.dump(label2id, writer)
    info('copy vocab')
    copyfile(vocab_path, os.path.join(save_dir, 'vocab.txt'))
