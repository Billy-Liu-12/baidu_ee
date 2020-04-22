# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 2:08 下午
# @Author  : lizhen
# @FileName: data_process.py
# @Description:
import json
import os
from transformers import BertTokenizer
import random
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import load_word_embedding, Wordembedding
import numpy as np
from sklearn.model_selection import train_test_split


class Argument():
    def __init__(self, argument, role, role_id, argument_start_index):
        self.argument = argument  # 论元
        self.role = role  # 论元对应的论元角色
        self.role_id = role_id  # 论元角色对应的id
        self.argument_start_index = argument_start_index  # 论元起始位置


class Event():
    def __init__(self, trigger, event_class, event_class_id, event_type, event_type_id, trigger_start_index):
        self.trigger = trigger  # 触发词
        self.event_class = event_class  # 事件类型对应的大类
        self.event_class_id = event_class_id  # 事件类型对应的大类id
        self.event_type = event_type  # 事件类型
        self.event_type_id = event_type_id  # 事件类型对应的id
        self.trigger_start_index = trigger_start_index  # 触发词的起始位置
        self.arguments = []
        self.text_seq_tags = []  # 序列标注的文本格式
        self.text_seq_tags_id = []  # 序列标注的id格式


class EESchema():
    def __init__(self, class2id, id2class, event2id, id2event, role2id, id2role):
        self.class2id = class2id
        self.id2class = id2class
        self.event2id = event2id
        self.id2event = id2event
        self.role2id = role2id
        self.id2role = id2role


class InputExample():
    def __init__(self, id, text):
        self.id = id  # 实例id
        self.text = text  # 事件描述
        self.events = []  # 事件
        self.text_token = []  # 对事件描述进行分词
        self.text_token_id = []  # 词对应词汇表中的id
        self.seq_tag_id = []  # 用于序列标注的label


class EEBertDataset(Dataset):
    def __init__(self, data_dir, file_name, schema_name, bert_path, device, valid_size=0.2,use_tag=False):
        self.device = device
        self.bert_path = bert_path
        self.data_dir = data_dir
        self.file_name = file_name
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.schema = self._load_schema(os.path.join(data_dir, schema_name))
        self.num_tag_labels = len(self.schema.role2id) * 2 + 1  # B I O
        if 'test' not in file_name:
            self.data = self._load_dataset()
            if use_tag:
                self.data = self.convert_data_tag()
            self.train_set,self.valid_set = train_test_split(self.data,test_size=valid_size,random_state=1024)


        else:
            self.data = self._load_testset()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def show_dataset_info(self):
        """该数据集的相关统计信息"""
        pass

    def convert_data_tag(self):
        """
        把数据转换成单句单事件的序列标注任务
        :return:
        """
        examples = []
        for data in self.data:
            text = data.text
            tokenized_text = data.text_token
            for e in data.events:
                arguments = {}
                for a in e.arguments:
                    index = a.role.rfind('-')
                    arguments[a.argument] = (a.role[:index], a.role[index + 1:])
                examples.append((data.text_token_id, e.text_seq_tags_id, text, arguments, tokenized_text))
        return examples

    def _load_testset(self):
        """
        加载测试集
        :return:
        """
        examples = []
        with open(os.path.join(self.data_dir, self.file_name)) as f:
            for l in f:
                l = json.loads(l)
                input_example = InputExample(l['id'], l['text'])
                input_example.text_token = self.tokenizer.tokenize(l['text'])
                input_example.text_token_id = [self.tokenizer.cls_token_id]+ self.tokenizer.encode(l['text'],add_special_tokens=False)

                examples.append(input_example)
        return examples

    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    def _load_dataset(self):
        """
        加载数据集：训练集，验证集
        :return:
        """
        examples = []
        if not os.path.exists(os.path.join(self.data_dir, self.file_name.split('.')[0] + '.pkl')):
            with open(os.path.join(self.data_dir, self.file_name)) as f:
                for l in tqdm(f):
                    l = json.loads(l)
                    input_example = InputExample(l['id'], l['text'])
                    # 对文本分词 以及 编码
                    input_example.text_token = self.tokenizer.tokenize(l['text'])
                    input_example.text_token_id = [self.tokenizer.cls_token_id] + self.tokenizer.encode(l['text'],add_special_tokens=False)

                    for e in l['event_list']:

                        event = Event(e['trigger'], e['class'], self.schema.class2id[e['class']], e['event_type'],
                                      self.schema.event2id[e['event_type']], e['trigger_start_index'])
                        text_seq_tag_id = [0] * len(input_example.text_token_id)
                        for a in e['arguments']:

                            argument = Argument(a['argument'], e['event_type'] + '-' + a['role'],
                                                self.schema.role2id[e['event_type'] + '-' + a['role']],
                                                a['argument_start_index'])

                            a_token_id = self.tokenizer.encode(a['argument'],add_special_tokens=False)
                            start_index = self.search(a_token_id, input_example.text_token_id)
                            if start_index != -1:
                                text_seq_tag_id[start_index] = self.schema.role2id[
                                                                   e['event_type'] + '-' + a['role']] * 2 + 1
                                for i in range(1, len(a_token_id)):
                                    text_seq_tag_id[start_index + i] = self.schema.role2id[
                                                                           e['event_type'] + '-' + a['role']] * 2 + 2

                            event.arguments.append(argument)
                        event.text_seq_tags_id = text_seq_tag_id[1:]
                        input_example.events.append(event)
                    examples.append(input_example)

            with open(os.path.join(self.data_dir, self.file_name.split('.')[0] + '.pkl'), 'wb') as f:
                pickle.dump(examples, f)
                print('saved pkl file')
        else:
            with open(os.path.join(self.data_dir, self.file_name.split('.')[0] + '.pkl'), 'rb') as f:
                examples = pickle.load(f)

        return examples

    def _load_schema(self, schema_filename_path):
        """

        :param schema_filename_path: schema 文件路径
        :return:
        """
        with open(schema_filename_path) as f:
            class2id, id2class, event2id, id2event, role2id, id2role = {}, {}, {}, {}, {}, {}
            class_idx, event_idx, role_idx = 0, 0, 0
            for l in f:
                l = json.loads(l)
                if l['class'] not in class2id:
                    class2id[l['class']] = class_idx
                    id2class[class_idx] = l['class']
                    class_idx += 1
                if l['event_type'] not in event2id:
                    event2id[l['event_type']] = event_idx
                    id2event[event_idx] = l['event_type']
                    event_idx += 1
                for role in l['role_list']:
                    if l['event_type'] + '-' + role['role'] not in role2id:
                        role2id[l['event_type'] + '-' + role['role']] = role_idx
                        id2role[role_idx] = l['event_type'] + '-' + role['role']
                        role_idx += 1

        return EESchema(class2id, id2class, event2id, id2event, role2id, id2role)


    def seq_tag_collate_fn(self, datas):
        """
        train phrase: 序列标注：对文本中的论元进行序列标注
        datas[0]: token_ids
        data[1]: seq_tags
        data[2];text
        data[3]:arugments
        data[4]:tokenized_text
        """
        seq_lens = []  # 记录每个句子的长度
        text_ids = []  # 训练数据text_token_id
        seq_tags = []  # 记录该batch下文本所对应的论元的序列标注
        texts = []
        arguments = []
        masks_bert = []
        masks_crf = []
        class_label = []
        event_label = []
        max_seq_len = len(max(datas, key=lambda x: len(x[0]))[0])  # 该batch中句子的最大长度
        for data in datas:
            seq_len = len(data[0])
            seq_lens.append(seq_len)
            mask_bert = [1] * seq_len
            mask_crf = [1] * (seq_len - 1)

            texts.append(data[4])
            arguments.append(data[3])
            if not data[3]:
                class_label.append(len(self.schema.class2id))
                event_label.append(len(self.schema.event2id))
            else:

                class_label.append(self.schema.class2id[list(data[3].values())[0][0].split('-')[0]])
                event_label.append(self.schema.event2id[list(data[3].values())[0][0]])
            text_ids.append(data[0] + [self.tokenizer.pad_token_id] * (max_seq_len - seq_len))  # 文本id
            masks_bert.append(mask_bert + [0] * (max_seq_len - seq_len))

            seq_tags.append(data[1] + [0] * (max_seq_len - seq_len))
            masks_crf.append(mask_crf + [0] * (max_seq_len - seq_len))

        text_ids = torch.LongTensor(np.array(text_ids)).to(self.device)
        seq_tags = torch.LongTensor(np.array(seq_tags)).to(self.device)
        masks_bert = torch.ByteTensor(np.array(masks_bert)).to(self.device)
        masks_crf = torch.ByteTensor(np.array(masks_crf)).to(self.device)
        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        class_label = torch.LongTensor(np.array(class_label)).to(self.device)
        event_label = torch.LongTensor(np.array(event_label)).to(self.device)
        return text_ids, seq_lens, masks_bert, masks_crf, texts, arguments,class_label,event_label, seq_tags

    def inference_collate_fn(self, datas):
        """
        test phrase: 序列标注：对文本中的论元进行序列标注
        """
        ids = []
        seq_lens = []  # 记录每个句子的长度
        text_ids = []  # 训练数据text_token_id
        texts = []  # 原文本
        masks_bert = []  # 统一长度，文本内容为1，非文本内容为0
        masks_crf = []
        max_seq_len = len(max(datas, key=lambda x: len(x.text_token_id)).text_token_id)  # 该batch中句子的最大长度
        for data in datas:
            ids.append(data.id)
            seq_len = len(data.text_token_id)  # 句子长度
            seq_lens.append(seq_len)
            mask_bert = [1] * seq_len
            mask_crf = [1] * (seq_len - 1)

            texts.append(data.text_token)
            text_ids.append(data.text_token_id + [self.tokenizer.pad_token_id] * (max_seq_len - seq_len))  # 文本id
            masks_bert.append(mask_bert + [0] * (max_seq_len - seq_len))
            masks_crf.append(mask_crf + [0] * (max_seq_len - seq_len))

        text_ids = torch.LongTensor(np.array(text_ids)).to(self.device)
        masks_bert = torch.ByteTensor(np.array(masks_bert)).to(self.device)
        masks_crf = torch.ByteTensor(np.array(masks_crf)).to(self.device)
        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        return ids, text_ids, seq_lens, masks_bert, masks_crf, texts


