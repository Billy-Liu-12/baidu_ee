import random
import sys
import gc
import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import os
import numpy as np
import pickle
import word2vec


def shuffle_aligned_list(data):
    """
        data是默认对齐的多个数据的list，
        [[text],[label]]
        每个元素是等长的列表
    """

    num = len(data[0])
    index = [i for i in range(num)]
    random.shuffle(index)
    return [[d[j] for j in index] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

        输入的data是默认对齐的多个数据的list，
        [[text],[label]]
        每个元素是等长的列表
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    data_count = 0
    data_len = len(data[0])
    while data_count < data_len:

        if data_count + batch_size > data_len:
            batch_size = data_len - data_count

        start = data_count
        end = start + batch_size
        data_count += batch_size

        dt = [d[start:end] for d in data]
        max_len = max([len(single) for single in dt[0]])  # 最大的句子长度

        for j in range(len(dt[0])):
            padding = [0] * (max_len - len(dt[0][j]))
            dt[0][j] += padding
            dt[1][j] += padding
            dt[2][j] += padding
            dt[3][j] += padding
            for key in dt[4][j]:
                dt[4][j][key] += padding
        # segments列表全0，因为只有一个句子1，没有句子2
        # input_masks列表1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
        # 相当于告诉BertModel不要利用后面0的部分
        yield dt


def test_batch_generator(data, batch_size=4, shuffle=True):
    """Generate batches of data.

        输入的data是默认对齐的多个数据的list，
        [[text],[label]]
        每个元素是等长的列表
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    data_count = 0
    data_len = len(data[0])
    while data_count < data_len:

        if data_count + batch_size > data_len:
            batch_size = data_len - data_count

        start = data_count
        end = start + batch_size
        data_count += batch_size

        dt = [d[start:end] for d in data]
        max_len = max([len(single) for single in dt[0]])  # 最大的句子长度

        for j in range(len(dt[0])):
            padding = [0] * (max_len - len(dt[0][j]))
            dt[0][j] += padding
            dt[1][j] += padding
            dt[2][j] += padding
        # segments列表全0，因为只有一个句子1，没有句子2
        # input_masks列表1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
        # 相当于告诉BertModel不要利用后面0的部分
        yield dt


def build_data(f_text, f_trigger_tag, f_argument_tag, tokenizer):
    texts = f_text.readlines()
    trigger_tag = f_trigger_tag.readlines()
    argument_tag = f_argument_tag.readlines()

    tokens, segments, input_masks = [], [], []

    for t in range(len(texts)):
        texts[t] = texts[t].replace('\n', '')
        # sen = '[CLS]' + texts[t]
        tokenized_text = ['[CLS]'] + list(texts[t])
        # tokenized_text = tokenizer.tokenize(sen)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    for t in range(len(trigger_tag)):
        trigger_tag[t] = [int(i) for i in trigger_tag[t].strip().split()]

    for a in range(len(argument_tag)):
        ag_ = {}
        ag_s = argument_tag[a].strip().split(';')
        for a_s in ag_s:
            if len(a_s) < 2:
                continue
            arg = [int(i) for i in a_s.strip().split()]
            ag_[arg[0]] = arg[1:]
        argument_tag[a] = ag_

        for i in range(len(texts)):
            if not len(tokens[i]) == (len(trigger_tag[i]) + 1):
                print(texts[i])
                tokenized_text = ['[CLS]'] + list(texts[i])
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                print(tokenized_text)
                print(len(tokenized_text))
                print(len(indexed_tokens))
                print(len(tokens[i]))
                print(len(trigger_tag[i]) + 1)
                print(i)
    data = [tokens, segments, input_masks, trigger_tag, argument_tag]

    return data


def build_test_data(f_text, tokenizer):
    texts = f_text.readlines()

    tokens, segments, input_masks = [], [], []

    for t in range(len(texts)):
        texts[t] = texts[t].replace('\n', '')
        # sen = '[CLS]' + texts[t]
        tokenized_text = ['[CLS]'] + list(texts[t])
        # tokenized_text = tokenizer.tokenize(sen)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    data = [tokens, segments, input_masks]

    return data


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class Wordembedding():
    def __init__(self, stoi, itos, vectors):
        self.stoi = stoi
        self.itos = itos
        self.vectors = vectors


def build_vocab(data_dir, train_file_name, valid_file_name, test_file_name, tokenizer=None):
    """
    构建词汇表
    """
    if not tokenizer:
        tokenizer = lambda x: [y for y in x]
    vocab = set()
    # 对训练集分词，更新词汇表
    with open(os.path.join(data_dir, train_file_name)) as f:
        for l in f:
            line = json.loads(l)
            vocab.update(tokenizer(line['text']))
    # 对验证集分词，更新词汇表
    with open(os.path.join(data_dir, valid_file_name)) as f:
        for l in f:
            line = json.loads(l)
            vocab.update(tokenizer(line['text']))
    # 对测试集分词，更新词汇表
    with open(os.path.join(data_dir, test_file_name)) as f:
        for l in f:
            line = json.loads(l)
            vocab.update(tokenizer(line['text']))
    vocab.add('UNK')
    vocab.add('PAD')
    return vocab


def build_word_embedding(vocab, embedding_size, word_embedding_path):
    vocab_size = len(vocab)
    vectors_ = np.random.normal(loc=0, scale=1, size=[vocab_size, embedding_size])
    pretrained_word_embedding = word2vec.load(word_embedding_path, kind="txt")
    stoi, itos = {}, {}
    for idx, char in enumerate(vocab):
        stoi[char] = idx
        itos[idx] = char
        if char in pretrained_word_embedding.vocab:
            vectors_[idx] = pretrained_word_embedding.vectors[pretrained_word_embedding.vocab_hash[char]]
    return Wordembedding(stoi, itos, vectors_)


def save_word_embedding(word_embedding_dir, embedding_name, new_word_embedding):
    with open('{}_{}.pkl'.format(word_embedding_dir, embedding_name), 'wb') as f:
        pickle.dump(new_word_embedding, f)


def load_word_embedding(word_embedding_path, embedding_name, data_dir=None, train_file_name=None, valid_file_name=None,
                        test_file_name=None, tokenizer=None):
    word_embedding_dir = word_embedding_path[:word_embedding_path.rfind('/')]
    if os.path.exists('{}_{}.pkl'.format(word_embedding_dir, embedding_name)):
        with open('{}_{}.pkl'.format(word_embedding_dir, embedding_name), 'rb') as f:
            word_embedding = pickle.load(f)
        return word_embedding
    else:
        vocab = build_vocab(data_dir, train_file_name, valid_file_name, test_file_name, tokenizer)
        new_word_embedding = build_word_embedding(vocab, embedding_size=300, word_embedding_path=word_embedding_path)
        save_word_embedding(word_embedding_dir=word_embedding_dir, embedding_name=embedding_name,
                            new_word_embedding=new_word_embedding)
        return new_word_embedding


def extract_arguments(text, pred_tag,schema):
    """arguments抽取函数
    """
    arguments, starting = [], False
    for i, label in enumerate(pred_tag):
        if label > 0:
            if label % 2 == 1:
                starting = True
                index = schema.id2role[(label - 1) // 2].rfind('-')
                arguments.append([[i], (
                schema.id2role[(label - 1) // 2][:index], schema.id2role[(label - 1) // 2][index + 1:])])
            elif starting:
                arguments[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False

    return {text[idx[0]:idx[-1]]: l for idx, l in arguments}
