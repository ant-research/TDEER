#! -*- coding:utf-8 -*-

import os
import argparse

from tokenizers import BertWordPieceTokenizer

from dataloader import DataGenerator, load_data, load_rel
from model import build_model, Evaluator
from utils import Infer, compute_metrics


parser = argparse.ArgumentParser(description='TDEER cli')
parser.add_argument('--do_train', action='store_true', help='to train TDEER, plz specify --do_train')
parser.add_argument('--do_test', action='store_true', help='specify --do_test to evaluate')
parser.add_argument('--model_name', type=str, required=True, help='specify the model name')
parser.add_argument('--rel_path', type=str, required=True, help='specify the relation path')
parser.add_argument('--train_path', type=str, help='specify the train path')
parser.add_argument('--dev_path', type=str, help='specify the dev path')
parser.add_argument('--test_path', type=str, help='specify the test path')
parser.add_argument('--bert_dir', type=str, help='specify the pre-trained bert model')
parser.add_argument('--save_path', default=None, type=str, help='specify the save path to save model [training phase]')
parser.add_argument('--ckpt_path', default=None, type=str, help='specify the ckpt path [test phase]')
parser.add_argument('--learning_rate', default=2e-5, type=float, help='specify the learning rate')
parser.add_argument('--epoch', default=100, type=int, help='specify the epoch size')
parser.add_argument('--batch_size', default=8, type=int, help='specify the batch size')
parser.add_argument('--max_len', default=120, type=int, help='specify the max len')
parser.add_argument('--neg_samples', default=None, type=int, help='specify negative sample num')
parser.add_argument('--max_sample_triples', default=None, type=int, help='specify max sample triples')
parser.add_argument('--verbose', default=2, type=int, help='specify verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch')
args = parser.parse_args()


print("Argument:", args)

id2rel, rel2id, all_rels = load_rel(args.rel_path)
tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_dir, 'vocab.txt'), lowercase=False)
tokenizer.enable_truncation(max_length=args.max_len)
entity_model, rel_model, translate_model, train_model = build_model(args.bert_dir, args.learning_rate, len(all_rels))

if args.do_train:
    assert args.save_path is not None, "please specify --save_path in traning phase"
    # check save
    train_model.save_weights(args.save_path)

    train_data = load_data(args.train_path, rel2id, is_train=True)
    dev_data = load_data(args.dev_path, rel2id, is_train=False)
    if args.test_path is not None:
        test_data = load_data(args.test_path, rel2id, is_train=False)
    else:
        test_data = None
    generator = DataGenerator(
        train_data, tokenizer, rel2id, all_rels,
        args.max_len, args.batch_size, args.max_sample_triples, args.neg_samples
    )
    infer = Infer(entity_model, rel_model, translate_model, tokenizer, id2rel)
    evaluator = Evaluator(infer, train_model, dev_data, args.save_path, args.model_name,
                          learning_rate=args.learning_rate)
    train_model.fit(
        generator.forfit(random=True),
        steps_per_epoch=len(generator),
        epochs=args.epoch,
        callbacks=[evaluator],
        verbose=args.verbose
    )

if args.do_test:
    assert args.ckpt_path is not None, "please specify --ckpt_path in test phase"
    test_data = load_data(args.test_path, rel2id, is_train=False)
    train_model.load_weights(args.ckpt_path)
    infer = Infer(entity_model, rel_model, translate_model, tokenizer, id2rel)
    precision, recall, f1_score = compute_metrics(infer, test_data, model_name=args.model_name)
    print(f'precision: {precision}, recall: {recall}, f1: {f1_score}')
