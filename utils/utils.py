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
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import jieba.posseg as pseg
import jieba


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


def bert_extract_arguments(text, pred_tag, schema, class_id, text_map_seg_idx, seg_idx_map_bert_idx,
                           bert_idx_map_seg_idx,seg_idx_map_text, raw_text):
    """arguments抽取函数
    """
    arguments, starting = [], False
    for i, label in enumerate(pred_tag):
        j = bert_idx_map_seg_idx[i]
        if label > 0:
            label = label + schema.move[class_id]
            if label % 2 == 1:
                starting = True
                index = schema.id2role[(label - 1) // 2].rfind('-')
                arguments.append(
                    [[j], (schema.id2role[(label - 1) // 2][:index], schema.id2role[(label - 1) // 2][index + 1:])])
            elif starting:
                arguments[-1][0].append(j)
            else:
                starting = False
        else:
            starting = False

    return {raw_text[seg_idx_map_text[idx[0]][0]:seg_idx_map_text[idx[-1]][1]].strip(): l for idx, l in arguments}


def extract_arguments(text, pred_tag, schema):
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
    return {text[idx[0]:idx[-1] + 1]: l for idx, l in arguments}


class Feature_example():
    def __init__(self, sentence_feature, text_ids, seq_lens, masks_bert, masks_crf, texts, arguments, class_labels,
                 event_labels, seq_tags):
        self.sentence_feature = sentence_feature
        self.text_ids = text_ids
        self.seq_lens = seq_lens
        self.masks_bert = masks_bert
        self.masks_crf = masks_crf
        self.texts = texts
        self.arguments = arguments
        self.class_labels = class_labels
        self.event_labels = event_labels
        self.seq_tags = seq_tags


class Test_example():
    def __init__(self, id, sf, text_id, seq_len, mask_bert, mask_crf, text):
        self.id = id
        self.sf = sf
        self.text_id = text_id
        self.seq_len = seq_len
        self.mask_bert = mask_bert
        self.mask_crf = mask_crf
        self.text = text


def inference_feature_collate_fn(datas):
    """
    使用bert抽取特征
    :param datas:
    :return:
    """
    ids = []
    sentence_feature = []
    text_ids = []
    seq_lens = []
    masks_bert = []
    masks_crf = []
    texts = []
    for t_example in datas:
        ids.append(t_example.id)
        sentence_feature.append(t_example.sf)
        text_ids.append(t_example.text_id)
        seq_lens.append(t_example.seq_len)
        masks_bert.append(t_example.mask_bert)
        masks_crf.append(t_example.mask_crf)
        texts.append(t_example.text)

    return ids, torch.from_numpy(np.array(sentence_feature)).cuda(), text_ids, torch.from_numpy(
        np.array(seq_lens)).cuda(), masks_bert, torch.from_numpy(np.array(masks_crf)).cuda(), texts


def feature_collate_fn(datas):
    """
    预训练的bert抽取其中特征

    :param datas:
    :return:
    """
    sentence_feature = []
    text_ids = []
    seq_lens = []
    masks_bert = []
    masks_crf = []
    texts = []
    arguments = []
    class_labels = []
    event_labels = []
    seq_tags = []

    for example in datas:
        sentence_feature.append(example.sentence_feature)
        text_ids.append(example.text_ids)
        seq_lens.append(example.seq_lens)
        masks_bert.append(example.masks_bert)
        masks_crf.append(example.masks_crf)
        texts.append(example.texts)
        arguments.append(example.arguments)
        class_labels.append(example.class_labels)
        event_labels.append(example.event_labels)
        seq_tags.append(example.seq_tags)

    return torch.from_numpy(np.array(sentence_feature)).cuda(), text_ids, torch.from_numpy(
        np.array(seq_lens)).cuda(), masks_bert, torch.from_numpy(
        np.array(masks_crf)).cuda(), texts, arguments, class_labels, event_labels, torch.from_numpy(
        np.array(seq_tags)).cuda()


def load_schema(schema_path):
    role2id, id2role = {}, {}
    role_idx = 0
    with open(schema_path) as f:
        for l in f:
            l = json.loads(l)
            for role in l['role_list']:
                if l['event_type'] + '-' + role['role'] not in role2id:
                    role2id[l['event_type'] + '-' + role['role']] = role_idx
                    id2role[role_idx] = l['event_type'] + '-' + role['role']
                    role_idx += 1
    return role2id, id2role


def load_raw_data(data_path, schema_path, modify_value=50, value=1e-6):
    """

    :param data_path: 文件路径
    :return:
    """
    role2id, id2role = load_schema(schema_path)
    start_transitions = np.zeros(435)
    end_transitions = np.zeros(435)
    transitions = np.zeros((435, 435))
    sample_count = 0
    with open(data_path, 'r') as f:
        for l in tqdm(f):
            sample_count += 1
            l = json.loads(l)
            argument_dict = defaultdict(set)

            for e in l['event_list']:

                for a in e['arguments']:
                    # 统计不重复论元
                    argument_dict[a['argument']].add(a['argument_start_index'])

                    # 对O的情况讨论：1.O->B， 对所有论元来说，先假设他们都是由 O 转移到 B 的情况，每遇到 以 B 开头或 B -> B 或 I-> B 的情况，对应的O->B的情况就减去1
                    transitions[0, role2id[e['event_type'] + '-' + a['role']] * 2 + 1] += 1

                    if a['argument_start_index'] == 0:  # 如果论元的起始位置为0，初始状态矩阵对应的论元的B标签的 值 + 1
                        start_transitions[role2id[e['event_type'] + '-' + a['role']] * 2 + 1] += 1
                        transitions[0, role2id[e['event_type'] + '-' + a['role']] * 2 + 1] -= 1

                    argument_end_index = a['argument_start_index'] + len(a['argument'])  # 论元结束下标
                    if argument_end_index == len(l['text']):  # 以论元结尾的情况
                        if len(a['argument']) == 1:  # 以B 结尾的情况
                            end_transitions[role2id[e['event_type'] + '-' + a['role']] * 2 + 1] += 1
                        else:  # 以I 结尾的情况
                            end_transitions[role2id[e['event_type'] + '-' + a['role']] * 2 + 2] += 1

                    # 状态转移矩阵

                    if len(a['argument']) == 1:
                        # 1. 以B 开头，以B 结束（即，论元长度为1,且下一个字仍然是论元的开头的情况）
                        for other_a in e['arguments']:
                            if other_a['argument_start_index'] == (a['argument_start_index'] + 1):
                                transitions[role2id[e['event_type'] + '-' + a['role']] * 2 + 1, role2id[
                                    e['event_type'] + '-' + other_a['role']] * 2 + 1] += 1
                                transitions[0, role2id[e['event_type'] + '-' + other_a['role']] * 2 + 1] -= 1

                                break

                        else:
                            # 2. 以B开头，下一个字符是 O的情况
                            transitions[role2id[e['event_type'] + '-' + a['role']] * 2 + 1, 0] += 1

                    else:
                        # 3. 以B开头，下一个字符是I 的情况
                        transitions[role2id[e['event_type'] + '-' + a['role']] * 2 + 1, role2id[
                            e['event_type'] + '-' + a['role']] * 2 + 2] += 1
                        # 1. I 转移到 I 的情况
                        transitions[role2id[e['event_type'] + '-' + a['role']] * 2 + 2, role2id[
                            e['event_type'] + '-' + a['role']] * 2 + 2] = (len(a['argument']) - 1)
                        # 2. I 转移到 B 的情况
                        for other_a in e['arguments']:
                            if other_a['argument_start_index'] == argument_end_index:
                                transitions[role2id[e['event_type'] + '-' + a['role']] * 2 + 2, role2id[
                                    e['event_type'] + '-' + other_a['role']] * 2 + 1] += 1
                                transitions[0, role2id[e['event_type'] + '-' + other_a['role']] * 2 + 1] -= 1

                                break
                        else:
                            # 3. I 转移到O的情况
                            transitions[role2id[e['event_type'] + '-' + a['role']] * 2 + 2, 0] += 1

            # 对O的情况讨论：2.O->O = 句子的长度 -（所有不重复论元的长度和）- （不重复论元个数：count）
            arguments_len = sum([len(k) * len(v) for k, v in argument_dict.items()])  # 所有不重复论元的长度和
            argument_count = sum([len(v) for k, v in argument_dict.items()])  # 不重复论元个数：count
            transitions[0, 0] = len(l['text']) - arguments_len - argument_count

        start_transitions[0] = sample_count - sum(start_transitions)
        end_transitions[0] = sample_count - sum(end_transitions)

        for i in range(217):
            # B->O:B转移到O的情况
            transitions[2 * i + 1, 0] += modify_value
            # I->O:I转移到O 的情况
            transitions[2 * i + 2, 0] += modify_value
            # -------讨论O的情况------------
            # O->B:O转移到当前B的情况
            transitions[0, 2 * i + 1] += modify_value
            # O->I:O转移到当前I的情况
            transitions[0, 2 * i + 2] += value

            for j in range(217):
                if i == j:
                    # -------讨论B的情况------------
                    # B->B:转移到自身的B
                    transitions[2 * i + 1, 2 * j + 1] += value
                    # B->I:转移到自身的I
                    transitions[2 * i + 1, 2 * j + 2] += modify_value
                    # -------讨论I的情况------------
                    # I->B:I转移到自身B的情况
                    transitions[2 * i + 2, 2 * j + 1] += value
                    # I->I:I转移到自身I的情况
                    transitions[2 * i + 2, 2 * j + 2] += modify_value
                else:
                    # -------讨论B的情况------------
                    # B->B:转移到其他的B
                    transitions[2 * i + 1, 2 * j + 1] += modify_value
                    # B->I:转移到其他的I
                    transitions[2 * i + 1, 2 * j + 2] += value
                    # -------讨论I的情况------------
                    # I->B:I转移到其他B的情况
                    transitions[2 * i + 2, 2 * j + 1] += modify_value
                    # I->I:I转移到其他I的情况
                    transitions[2 * i + 2, 2 * j + 2] += value

            transitions[i] = np.nan_to_num((np.divide(transitions[i], sum(transitions[i]))))

        start_transitions = np.nan_to_num((np.divide(start_transitions, sum(start_transitions))))
        end_transitions = np.nan_to_num((np.divide(end_transitions, sum(end_transitions))))

    return start_transitions, transitions, end_transitions


def get_restrain_crf(device, class_tag_num):
    restrain = []
    restrain_start = np.zeros(class_tag_num)
    restrain_end = np.zeros(class_tag_num)
    restrain_trans = np.zeros((class_tag_num, class_tag_num))

    for i in range(0, class_tag_num):
        if i > 0 and i % 2 == 0:
            restrain_start[i] = -1000.0
        if i == 0:
            for j in range(0, class_tag_num):
                if j > 0 and j % 2 == 0:
                    restrain_trans[i][j] = -1000.0
        if i % 2 == 1:
            for j in range(0, class_tag_num):
                if j > 0 and j % 2 == 0 and not j == (i + 1):
                    restrain_trans[i][j] = -1000.0

    restrain.append(torch.from_numpy(restrain_start).float().cuda())
    restrain.append(torch.from_numpy(restrain_end).float().cuda())
    restrain.append(torch.from_numpy(restrain_trans).float().cuda())
    return restrain


def postagger(txt):
    words = pseg.cut(txt)

    wdflags = [b for a, b in words]

    return wdflags


def merge_result():
    res = defaultdict(list)
    for i in range(9):
        with open('../data/tag_result/result_{}.json'.format(i), 'r') as f:
            for l in f:
                l = json.loads(l)
                res[(l['id'], l['text'])].extend(l['event_list'])

    with open('../result.json', 'w') as f:
        for k, v in res.items():
            temp_res = defaultdict(list)
            temp_list = []
            for e in v:
                temp_res[e['event_type']].extend(e['arguments'])
            for temp_k, temp_v in temp_res.items():
                temp_list.append({'event_type': temp_k, 'arguments': temp_v})

            f.write(json.dumps({'id': k[0], 'text': k[1], 'event_list': temp_list}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    merge_result()
    # print(''.join(map(lambda x: str(x), [1, 2, 3])).find(str(2)))
