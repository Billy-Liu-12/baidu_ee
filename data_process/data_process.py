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
import numpy as np
from sklearn.model_selection import train_test_split
import jieba
import numpy as np
from pyltp import SentenceSplitter, Postagger, NamedEntityRecognizer, Parser
import copy


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
        self.move = {0: 0, 1: 66, 2: 100, 3: 132, 4: 170, 5: 238, 6: 288, 7: 360, 8: 406}
        self.event_move = {0: 0, 1: 9, 2: 14, 3: 19, 4: 25, 5: 37, 6: 45, 7: 53, 8: 61}
        self.class_tag_num = {0: 67, 1: 35, 2: 33, 3: 39, 4: 69, 5: 51, 6: 73, 7: 47, 8: 29}
        self.event_num = {0: 9, 1: 5, 2: 5, 3: 6, 4: 12, 5: 8, 6: 8, 7: 8, 8: 4}


class InputExample():
    def __init__(self, id, text):
        self.id = id  # 实例id
        self.text = text  # 事件描述
        self.text_seg_tag = []
        self.text_pos_tag = []
        self.text_ner_tag = []
        self.text_map_seg_idx = []
        self.seg_idx_map_bert_idx = []
        self.events = []  # 事件
        self.text_token = []  # 对事件描述进行分词
        self.text_token_id = []  # 词对应词汇表中的id
        self.seq_tag_id = []  # 用于序列标注的label
        self.bert_idx_map_seg_idx = []
        self.seg_idx_map_text = []
        self.adj_matrix_subject = []
        self.adj_matrix_others = []
        self.adj_matrix_object = []
        self.kernal_verb = []


class TextClsDataset(Dataset):
    def __init__(self, data_dir, file_name, schema_name, bert_path, device, valid_size=0.2):
        self.device = device
        self.bert_path = bert_path
        self.data_dir = data_dir
        self.file_name = file_name
        self.seg_tag_set = set()
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.schema = self._load_schema(os.path.join(data_dir, schema_name))

        if 'test' not in file_name:
            self.data = self._load_dataset()
            self.data = self.convert_data_single_sentence_multi_event()
            self.convert_set_tag_2_dict_tag()
            self.train_set, self.valid_set = train_test_split(self.data, test_size=valid_size, random_state=13)
        else:
            self.convert_set_tag_2_dict_tag()
            self.data = self._load_testset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _load_testset(self):
        examples = []
        with open(os.path.join(self.data_dir, self.file_name)) as f:
            for l in tqdm(f):
                l = json.loads(l)
                # 分词
                text_seg = jieba.lcut(l['text'], HMM=False)
                input_example = InputExample(id=l['id'], text=l['text'])

                bert_tokens, text_map_seg_idx, seg_idx_map_bert_idx = self.text_map_bert(text_seg)
                seg_tag_align_bert = self.align_segword_2_bert(seg_idx_map_bert_idx)

                assert len(bert_tokens) - 1 == len(seg_tag_align_bert)

                input_example.text_token = bert_tokens
                input_example.text_token_id = self.tokenizer.encode(bert_tokens, add_special_tokens=False)
                input_example.text_seg_tag = seg_tag_align_bert
                input_example.seg_idx_map_bert_idx = seg_idx_map_bert_idx
                input_example.text_map_seg_idx = text_map_seg_idx
                examples.append(input_example)
        return examples

    def _load_dataset(self):
        """
        加载数据集：训练集，验证集
        :return:
        """
        examples = []

        with open(os.path.join(self.data_dir, self.file_name)) as f:
            for l in tqdm(f):
                l = json.loads(l)
                # 分词 pos ner : 中文命名实体识别是字符级模型（bert），所以用 list将字符串转换为字符列表。至于输出，格式为 (entity, type, begin, end)。
                text_seg = jieba.lcut(l['text'], HMM=False)
                example = self.align_bert(l, text_seg)
                examples.append(example)
        return examples

    def convert_data_single_sentence_multi_event(self):
        examples = []
        for data in self.data:
            text = data.text
            tokenized_text = data.text_token
            self.seg_tag_set.update(data.text_seg_tag)
            class_label = [0] * 9
            for idx, e in enumerate(data.events):
                class_label[self.schema.class2id[e.event_class]] = 1
            examples.append((data.text_token_id, text, tokenized_text, class_label))
        return examples

    def convert_set_tag_2_dict_tag(self):

        if not os.path.exists(os.path.join(self.data_dir, 'seg_tag.pkl')):
            self.seg_tag_dict = {}
            self.seg_tag_dict['MASK'] = 0
            for elem in self.seg_tag_set:
                self.seg_tag_dict[elem] = len(self.seg_tag_dict)
            with open(os.path.join(self.data_dir, 'seg_tag.pkl'), 'wb') as f:
                pickle.dump(self.seg_tag_dict, f)
                print('saved seg_tag_dict pkl file')
        else:
            with open(os.path.join(self.data_dir, 'seg_tag.pkl'), 'rb') as f:
                self.seg_tag_dict = pickle.load(f)

    def align_bert(self, l, text_seg):
        """"""
        input_example = InputExample(id=l['id'], text=l['text'])

        bert_tokens, text_map_seg_idx, seg_idx_map_bert_idx = self.text_map_bert(text_seg)
        seg_tag_align_bert = self.align_segword_2_bert(seg_idx_map_bert_idx)

        assert len(bert_tokens) - 1 == len(seg_tag_align_bert)

        input_example.text_token = bert_tokens
        input_example.text_token_id = self.tokenizer.encode(bert_tokens, add_special_tokens=False)
        input_example.text_seg_tag = seg_tag_align_bert
        input_example.seg_idx_map_bert_idx = seg_idx_map_bert_idx
        input_example.text_map_seg_idx = text_map_seg_idx
        for e in l['event_list']:
            event = Event(e['trigger'], e['class'], self.schema.class2id[e['class']], e['event_type'],
                          self.schema.event2id[e['event_type']], e['trigger_start_index'])
            input_example.events.append(event)
        return input_example

    def align_segword_2_bert(self, seg_idx_map_bert_idx):
        """
        采用 B、M、E、S 标记
        :param seg_idx_map_bert_idx: [seg_word1,seg_word2,....] ,seg_word=(start,end),[(1,4),(4,5),(5,8),()]
        :return:
        """
        seg_tag_align_bert = []
        for seg_tuple in seg_idx_map_bert_idx:
            start, end = seg_tuple
            if end == start:
                continue
            if end - start == 1:
                seg_tag_align_bert.append('S')
            else:
                seg_tag_align_bert.append('B')
                seg_tag_align_bert.extend(['M'] * (end - start - 2))
                seg_tag_align_bert.append('E')
        return seg_tag_align_bert

    def text_map_bert(self, text_seg):
        # text idx map seg idx,seg idx map bert idx range
        """
        1. 从文本id 映射到-> 分词后的词的id:   [1,1,2,2,2,3,3]-->[1,2,3]
        2. 词的id 映射到-> bert 分词后的范围(start,end)  [1,2,3,4]-->[(1,4),(4,5),(5,8),()]
        :param text_seg:
        :return:
        """
        text_map_seg_idx = []
        bert_tokens = ['[CLS]']
        seg_idx_map_bert_idx = []
        for seg_idx, token in enumerate(text_seg):
            # 从文本id 映射到-> 分词后的词的id
            text_map_seg_idx.extend([seg_idx] * len(token))

            # bert 对分词后的词再分词
            # bert_token = self.tokenizer.tokenize(token.replace(' ','[UNK]').replace('\n','[UNK]').replace('\r','[UNK]'))
            bert_token = self.tokenizer.tokenize(token)
            # 原分词的id 映射到-> bert 分词后的范围(start,end)  不包含end
            seg_idx_map_bert_idx.append((len(bert_tokens), len(bert_tokens) + len(bert_token)))
            # 存放bert分词结果
            bert_tokens.extend(bert_token)
        return bert_tokens, text_map_seg_idx, seg_idx_map_bert_idx

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

    def multi_event_seq_tag_collate_fn(self, datas):
        """
        train phrase: 序列标注：对文本中的论元进行序列标注
        datas[0]: token_ids
        data[1]: text
        data[2]:tokenized_text
        data[3]:class_label
        (data.text_token_id, text, tokenized_text,class_label)
        """
        seq_lens = []  # 记录每个句子的长度
        text_ids = []  # 训练数据text_token_id
        seq_tags = []  # 记录该batch下文本所对应的论元的序列标注
        texts = []
        masks_bert = []
        class_labels = []
        max_seq_len = len(max(datas, key=lambda x: len(x[0]))[0])  # 该batch中句子的最大长度
        for data in datas:
            seq_len = len(data[0])
            seq_lens.append(seq_len)
            mask_bert = [1] * seq_len
            class_labels.append(data[3])

            texts.append(data[1])

            text_ids.append(data[0] + [self.tokenizer.pad_token_id] * (max_seq_len - seq_len))  # 文本id
            masks_bert.append(mask_bert + [0] * (max_seq_len - seq_len))

            # seq_tags.append(data[1] + [0] * (max_seq_len - seq_len))

        text_ids = torch.LongTensor(np.array(text_ids)).to(self.device)
        seq_tags = torch.LongTensor(np.array(seq_tags)).to(self.device)
        masks_bert = torch.ByteTensor(np.array(masks_bert)).to(self.device)
        # seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        class_labels = torch.FloatTensor(np.array(class_labels)).to(self.device)

        return text_ids, seq_lens, masks_bert, texts, class_labels

    def inference_collate_fn(self, datas):
        """
        test phrase: 序列标注：对文本中的论元进行序列标注
        data 为input_example 对象
        """
        ids = []
        seq_lens = []  # 记录每个句子的长度
        text_ids = []  # 训练数据text_token_id
        texts_token = []  # 分词后的文本
        masks_bert = []  # 统一长度，文本内容为1，非文本内容为0
        texts = []

        max_seq_len = len(max(datas, key=lambda x: len(x.text_token_id)).text_token_id)  # 该batch中句子的最大长度
        for data in datas:
            ids.append(data.id)
            texts.append(data.text)
            seq_len = len(data.text_token_id)  # 句子长度
            seq_lens.append(seq_len)
            mask_bert = [1] * seq_len

            texts_token.append(data.text_token[1:])
            text_ids.append(data.text_token_id + [self.tokenizer.pad_token_id] * (max_seq_len - seq_len))  # 文本id
            masks_bert.append(mask_bert + [0] * (max_seq_len - seq_len))

        text_ids = torch.LongTensor(np.array(text_ids)).to(self.device)
        masks_bert = torch.ByteTensor(np.array(masks_bert)).to(self.device)
        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        return ids, text_ids, seq_lens, masks_bert, texts_token, texts


################################################################################################################################################################


########################################################################################################################################################################################################

class EEBertDataset(Dataset):
    def __init__(self, data_dir, file_name, schema_name, bert_path, device, class_id, valid_size=0.2):
        self.device = device
        self.bert_path = bert_path
        self.data_dir = data_dir
        self.file_name = file_name
        self.class_id = class_id
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.ltp_dir = '/data/lz/workspace/EE/data/ltp_model'
        self.verb = {'SBV',
                     'VOB',
                     'IOB',
                     'FOB',
                     'CMP'}
        self.seg_tag_set = set()
        self.pos_tag_set = set()
        self.ner_tag_set = set()
        if 'test' not in file_name:
            self.schema = self._load_schema(os.path.join(data_dir, schema_name))
            self.num_tag_labels = len(self.schema.role2id) * 2 + 1  # B I O
            self.data = self._load_dataset()
            self.data = self.convert_data_single_sentence_single_event()
            self.convert_set_tag_2_dict_tag()
            self.train_set, self.valid_set = train_test_split(self.data, test_size=valid_size, random_state=13)
        else:
            self.schema = self._load_schema(schema_name)
            self.num_tag_labels = len(self.schema.role2id) * 2 + 1  # B I O
            self.convert_set_tag_2_dict_tag()
            self.data = self._load_testset()

    def convert_set_tag_2_dict_tag(self):

        if not os.path.exists(os.path.join('data/raw_data', 'seg_tag.pkl')):
            self.seg_tag_dict = {}
            self.seg_tag_dict['MASK'] = 0
            for elem in self.seg_tag_set:
                self.seg_tag_dict[elem] = len(self.seg_tag_dict)
            with open(os.path.join('data/raw_data', 'seg_tag.pkl'), 'wb') as f:
                pickle.dump(self.seg_tag_dict, f)
                print('saved seg_tag_dict pkl file')
        else:
            with open(os.path.join('data/raw_data', 'seg_tag.pkl'), 'rb') as f:
                self.seg_tag_dict = pickle.load(f)

        # if not os.path.exists(os.path.join(self.data_dir, 'pos_tag.pkl')):
        #     self.pos_tag_dict = {}
        #     self.pos_tag_dict['MASK'] = 0
        #     for elem in self.pos_tag_set:
        #         self.pos_tag_dict[elem] = len(self.pos_tag_dict)
        #     with open(os.path.join(self.data_dir, 'pos_tag.pkl'), 'wb') as f:
        #         pickle.dump(self.pos_tag_dict, f)
        #         print('saved pos_tag_dict pkl file')
        # else:
        #     with open(os.path.join(self.data_dir, 'pos_tag.pkl'), 'rb') as f:
        #         self.pos_tag_dict = pickle.load(f)
        #
        # if not os.path.exists(os.path.join(self.data_dir, 'ner_tag.pkl')):
        #     self.ner_tag_dict = {}
        #     self.ner_tag_dict['MASK'] = 0
        #     for elem in self.ner_tag_set:
        #         self.ner_tag_dict[elem] = len(self.ner_tag_dict)
        #     with open(os.path.join(self.data_dir, 'ner_tag.pkl'), 'wb') as f:
        #         pickle.dump(self.ner_tag_dict, f)
        #         print('saved ner_tag_dict pkl file')
        # else:
        #     with open(os.path.join(self.data_dir, 'ner_tag.pkl'), 'rb') as f:
        #         self.ner_tag_dict = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def show_dataset_info(self):
        """该数据集的相关统计信息"""
        pass

    def convert_data_single_sentence_single_event(self):
        examples = []
        for data in self.data:
            text = data.text
            tokenized_text = data.text_token
            self.seg_tag_set.update(data.text_seg_tag)

            arguments = {}
            for e in data.events:
                seq_tag = copy.deepcopy(e.text_seq_tags_id)
                for a in e.arguments:
                    index = a.role.rfind('-')
                    arguments[a.argument + "_" + a.role] = (a.role[:index], a.role[index + 1:])

                begin = 0
                for i in range(len(seq_tag)):
                    if seq_tag[i] % 2 == 1:
                        begin = seq_tag[i]
                        continue
                    if seq_tag[i] > 0:
                        if not (seq_tag[i] - begin) == 1:
                            seq_tag[i] = seq_tag[i] - 1
                            begin = seq_tag[i]
                            continue
                    if seq_tag[i] == 0:
                        begin = 0

                for i in range(len(seq_tag)):
                    if seq_tag[i]:
                        seq_tag[i] = seq_tag[i] - self.schema.move[self.class_id]

                examples.append((data.text_token_id, seq_tag, text, arguments, tokenized_text, data.text_seg_tag,
                                 data.text_map_seg_idx, data.seg_idx_map_bert_idx, data.bert_idx_map_seg_idx,
                                 data.seg_idx_map_text, data.adj_matrix_subject, data.adj_matrix_object,
                                 data.adj_matrix_others, data.kernal_verb, e.event_type_id-self.schema.event_move[self.class_id]))
        return examples

    def convert_data_single_sentence_multi_event(self):
        examples = []
        for data in self.data:
            text = data.text
            tokenized_text = data.text_token
            self.seg_tag_set.update(data.text_seg_tag)

            text_seq_tags = []
            arguments = {}
            for e in data.events:
                for a in e.arguments:
                    index = a.role.rfind('-')
                    arguments[a.argument + "_" + a.role] = (a.role[:index], a.role[index + 1:])
                text_seq_tags.append(e.text_seq_tags_id)

            seq_tag = text_seq_tags[0]
            if len(text_seq_tags) > 1:
                for i in range(len(text_seq_tags[0])):  # col
                    for j in range(1, len(text_seq_tags)):  # row
                        seq_tag[i] = text_seq_tags[j][i] or seq_tag[i]
            begin = 0
            for i in range(len(seq_tag)):
                if seq_tag[i] % 2 == 1:
                    begin = seq_tag[i]
                    continue
                if seq_tag[i] > 0:
                    if not (seq_tag[i] - begin) == 1:
                        seq_tag[i] = seq_tag[i] - 1
                        begin = seq_tag[i]
                        continue
                if seq_tag[i] == 0:
                    begin = 0

            for i in range(len(seq_tag)):
                if seq_tag[i]:
                    seq_tag[i] = seq_tag[i] - self.schema.move[self.class_id]

            examples.append((data.text_token_id, seq_tag, text, arguments, tokenized_text, data.text_seg_tag,
                             data.text_map_seg_idx, data.seg_idx_map_bert_idx, data.bert_idx_map_seg_idx,
                             data.seg_idx_map_text, data.adj_matrix_subject, data.adj_matrix_object,
                             data.adj_matrix_others, data.kernal_verb))
        return examples

    def _load_testset(self):
        """
        加载测试集
        :return:
        """
        par_model_path = os.path.join(self.ltp_dir, 'parser.model')
        pos_model_path = os.path.join(self.ltp_dir, 'pos.model')
        postagger = Postagger()
        postagger.load(pos_model_path)
        parser = Parser()
        parser.load(par_model_path)

        examples = []
        with open(os.path.join(self.data_dir, self.file_name)) as f:
            for l in tqdm(f):
                l = json.loads(l)
                # 分词 pos ner : 中文命名实体识别是字符级模型（bert），所以用 list将字符串转换为字符列表。至于输出，格式为 (entity, type, begin, end)。
                text_seg = jieba.lcut(l['text'], HMM=False)
                poses = ' '.join(postagger.postag(text_seg)).split()
                arcs = parser.parse(text_seg, poses)
                arcses = ' '.join(
                    "%d:%s" % (arc.head, arc.relation) for arc in arcs).split()
                examples.append(self.align_bert_4_inference(l, text_seg, arcses))

        return examples

    def _load_dataset(self):
        """
        加载数据集：训练集，验证集
        :return:
        """
        par_model_path = os.path.join(self.ltp_dir, 'parser.model')
        pos_model_path = os.path.join(self.ltp_dir, 'pos.model')
        postagger = Postagger()
        postagger.load(pos_model_path)
        parser = Parser()
        parser.load(par_model_path)

        examples = []
        if not os.path.exists(
                os.path.join(self.data_dir, self.file_name.split('.')[0] + '_{}_.pkl'.format(self.class_id))):
            with open(os.path.join(self.data_dir, self.file_name)) as f:
                for l in tqdm(f):
                    l = json.loads(l)
                    # 分词 pos ner : 中文命名实体识别是字符级模型（bert），所以用 list将字符串转换为字符列表。至于输出，格式为 (entity, type, begin, end)。
                    text_seg = jieba.lcut(l['text'], HMM=False)
                    poses = ' '.join(postagger.postag(text_seg)).split()
                    arcs = parser.parse(text_seg, poses)
                    arcses = ' '.join(
                        "%d:%s" % (arc.head, arc.relation) for arc in arcs).split()

                    example = self.align_bert(l, text_seg, arcses)
                    if len(example.events) == 0:
                        continue
                    examples.append(example)
            with open(os.path.join(self.data_dir, self.file_name.split('.')[0] + '_{}_.pkl'.format(self.class_id)),
                      'wb') as f:
                pickle.dump(examples, f)
                print('saved {}'.format(
                    os.path.join(self.data_dir, self.file_name.split('.')[0] + '_{}_.pkl'.format(self.class_id))))
        else:
            with open(os.path.join(self.data_dir, self.file_name.split('.')[0] + '_{}_.pkl'.format(self.class_id)),
                      'rb') as f:
                examples = pickle.load(f)

        return examples

    def text_map_bert(self, text_seg):
        # text idx map seg idx,seg idx map bert idx range
        """
        1. 从文本id 映射到-> 分词后的词的id:   [1,1,2,2,2,3,3]-->[1,2,3]
        2. 词的id 映射到-> bert 分词后的范围(start,end)  [1,2,3,4]-->[(1,4),(4,5),(5,8),()]
        :param text_seg:
        :return:
        """
        text_map_seg_idx = []
        seg_idx_map_text = {}
        bert_tokens = ['[CLS]']
        seg_idx_map_bert_idx = []

        bert_idx_map_seg_idx = []
        for seg_idx, token in enumerate(text_seg):
            # 从文本id 映射到-> 分词后的词的id

            seg_idx_map_text[seg_idx] = (len(text_map_seg_idx), len(text_map_seg_idx) + len(token))
            text_map_seg_idx.extend([seg_idx] * len(token))
            # bert 对分词后的词再分词
            # bert_token = self.tokenizer.tokenize(token.replace(' ','[UNK]').replace('\n','[UNK]').replace('\r','[UNK]'))
            bert_token = self.tokenizer.tokenize(token)
            # 原分词的id 映射到-> bert 分词后的范围(start,end)  不包含end
            seg_idx_map_bert_idx.append((len(bert_tokens), len(bert_tokens) + len(bert_token)))
            # 存放bert分词结果
            bert_tokens.extend(bert_token)
            bert_idx_map_seg_idx.extend([seg_idx] * len(bert_token))

        return bert_tokens, text_map_seg_idx, seg_idx_map_bert_idx, bert_idx_map_seg_idx, seg_idx_map_text

    def align_segword_2_bert(self, seg_idx_map_bert_idx):
        """
        采用 B、M、E、S 标记
        :param seg_idx_map_bert_idx: [seg_word1,seg_word2,....] ,seg_word=(start,end),[(1,4),(4,5),(5,8),()]
        :return:
        """
        seg_tag_align_bert = []
        for seg_tuple in seg_idx_map_bert_idx:
            start, end = seg_tuple
            if end == start:
                continue
            if end - start == 1:
                seg_tag_align_bert.append('S')
            else:
                seg_tag_align_bert.append('B')
                seg_tag_align_bert.extend(['M'] * (end - start - 2))
                seg_tag_align_bert.append('E')
        return seg_tag_align_bert

    def align_pos_2_bert(self, text_pos, seg_idx_map_bert_idx):
        """

        :param text_pos:
        :param seg_idx_map_bert_idx:
        :return:
        """
        pos_tag_align_bert = []
        for pos, seg_tuple in zip(text_pos, seg_idx_map_bert_idx):
            start, end = seg_tuple
            pos_tag_align_bert.extend([pos] * (end - start))
        return pos_tag_align_bert

    def align_ner_2_bert(self, text_ner, bert_tokens, text_map_seg_idx, seg_idx_map_bert_idx):
        """

        :param text_ner:
        :param text_map_seg_idx:
        :param seg_idx_map_bert_idx:
        :return:
        """
        ner_tag_align_bert = ['O'] * len(bert_tokens)
        for ner in text_ner:
            # (entity, type, start, end)。
            ner_type = ner[1]
            start = ner[2]
            end = ner[3]
            bert_start_idx = seg_idx_map_bert_idx[text_map_seg_idx[start]][0]
            bert_end_idx = seg_idx_map_bert_idx[text_map_seg_idx[end - 1]][1]
            ner_tag_align_bert[bert_start_idx:bert_end_idx] = [ner_type] * (bert_end_idx - bert_start_idx)

        return ner_tag_align_bert[1:]

    def align_bert_4_inference(self, l, text_seg, arcses):
        input_example = InputExample(id=l['id'], text=l['text'])

        bert_tokens, text_map_seg_idx, seg_idx_map_bert_idx, bert_idx_map_seg_idx, seg_idx_map_text = self.text_map_bert(
            text_seg)
        seg_tag_align_bert = self.align_segword_2_bert(seg_idx_map_bert_idx)

        adj_matrix_object = np.zeros((len(bert_tokens), len(bert_tokens)))
        adj_matrix_subject = np.zeros((len(bert_tokens), len(bert_tokens)))
        adj_matrix_others = np.zeros((len(bert_tokens), len(bert_tokens)))
        obj_relation = {
            'VOB',
            'IOB',
            'FOB'
        }
        adj_matrix_back = np.zeros((len(bert_tokens), len(bert_tokens)))

        kernal_verb = [0] * len(bert_tokens)
        for idx, a in enumerate(arcses):
            if not (int(a.split(':')[0]) - 1) == 0:
                b_father, e_father = seg_idx_map_bert_idx[int(a.split(':')[0]) - 1]
                b_son, e_son = seg_idx_map_bert_idx[idx]
                if b_father == e_father or b_son == e_son:
                    continue
                adj_matrix_others[b_father:e_father, b_son:e_son] = 1 / (e_son - b_son)

                if a.split(':')[1] in obj_relation:
                    adj_matrix_object[b_son:e_son, b_father:e_father] = 1 / (e_father - b_father)
                elif a.split(':')[1] == 'SBV':
                    adj_matrix_subject[b_son:e_son, b_father:e_father] = 1 / (e_father - b_father)
                else:
                    adj_matrix_others[b_son:e_son, b_father:e_father] = 1 / (e_father - b_father)

                if a.split(':')[1] in self.verb:
                    kernal_verb[int(a.split(':')[0]) - 1] = 1
                if a.split(':')[1] == 'HED':
                    kernal_verb[idx] = 1

        assert len(bert_tokens) - 1 == len(seg_tag_align_bert)

        input_example.text_token = bert_tokens
        input_example.text_token_id = self.tokenizer.encode(bert_tokens, add_special_tokens=False)
        input_example.text_seg_tag = seg_tag_align_bert
        input_example.seg_idx_map_bert_idx = seg_idx_map_bert_idx
        input_example.text_map_seg_idx = text_map_seg_idx
        input_example.bert_idx_map_seg_idx = bert_idx_map_seg_idx
        input_example.seg_idx_map_text = seg_idx_map_text
        input_example.adj_matrix_subject = adj_matrix_subject
        input_example.adj_matrix_object = adj_matrix_object
        input_example.adj_matrix_others = adj_matrix_others
        input_example.kernal_verb = kernal_verb

        return input_example

    def align_bert(self, l, text_seg, arcses):
        """"""
        input_example = InputExample(id=l['id'], text=l['text'])

        bert_tokens, text_map_seg_idx, seg_idx_map_bert_idx, bert_idx_map_seg_idx, seg_idx_map_text = self.text_map_bert(
            text_seg)
        seg_tag_align_bert = self.align_segword_2_bert(seg_idx_map_bert_idx)

        adj_matrix_object = np.zeros((len(bert_tokens), len(bert_tokens)))
        adj_matrix_subject = np.zeros((len(bert_tokens), len(bert_tokens)))
        adj_matrix_others = np.zeros((len(bert_tokens), len(bert_tokens)))
        obj_relation = {
            'VOB',
            'IOB',
            'FOB'
        }
        adj_matrix_back = np.zeros((len(bert_tokens), len(bert_tokens)))

        kernal_verb = [0] * len(bert_tokens)
        for idx, a in enumerate(arcses):
            if not (int(a.split(':')[0]) - 1) == 0:
                b_father, e_father = seg_idx_map_bert_idx[int(a.split(':')[0]) - 1]
                b_son, e_son = seg_idx_map_bert_idx[idx]
                if b_father == e_father or b_son == e_son:
                    continue
                adj_matrix_others[b_father:e_father, b_son:e_son] = 1 / (e_son - b_son)

                if a.split(':')[1] in obj_relation:
                    adj_matrix_object[b_son:e_son, b_father:e_father] = 1 / (e_father - b_father)
                elif a.split(':')[1] == 'SBV':
                    adj_matrix_subject[b_son:e_son, b_father:e_father] = 1 / (e_father - b_father)
                else:
                    adj_matrix_others[b_son:e_son, b_father:e_father] = 1 / (e_father - b_father)

                if a.split(':')[1] in self.verb:
                    kernal_verb[int(a.split(':')[0]) - 1] = 1
                if a.split(':')[1] == 'HED':
                    kernal_verb[idx] = 1

        assert len(bert_tokens) - 1 == len(seg_tag_align_bert)

        input_example.text_token = bert_tokens
        input_example.text_token_id = self.tokenizer.encode(bert_tokens, add_special_tokens=False)
        input_example.text_seg_tag = seg_tag_align_bert
        input_example.seg_idx_map_bert_idx = seg_idx_map_bert_idx
        input_example.text_map_seg_idx = text_map_seg_idx
        input_example.bert_idx_map_seg_idx = bert_idx_map_seg_idx
        input_example.seg_idx_map_text = seg_idx_map_text
        input_example.adj_matrix_subject = adj_matrix_subject
        input_example.adj_matrix_object = adj_matrix_object
        input_example.adj_matrix_others = adj_matrix_others
        input_example.kernal_verb = kernal_verb
        for e in l['event_list']:
            if not self.schema.class2id[e['class']] == self.class_id:
                continue
            event = Event(e['trigger'], e['class'], self.schema.class2id[e['class']], e['event_type'],
                          self.schema.event2id[e['event_type']], e['trigger_start_index'])
            text_seq_tag_id = [0] * len(input_example.text_token_id)
            for a in e['arguments']:
                argument = Argument(a['argument'], e['event_type'] + '-' + a['role'],
                                    self.schema.role2id[e['event_type'] + '-' + a['role']],
                                    a['argument_start_index'])
                argument_start_index = a['argument_start_index']
                argument_end_index = a['argument_start_index'] + len(a['argument'])
                bert_token_start_index = seg_idx_map_bert_idx[text_map_seg_idx[argument_start_index]][0]
                bert_token_end_index = seg_idx_map_bert_idx[text_map_seg_idx[argument_end_index - 1]][1]

                text_seq_tag_id[bert_token_start_index] = self.schema.role2id[
                                                              e['event_type'] + '-' + a['role']] * 2 + 1
                text_seq_tag_id[(bert_token_start_index + 1):bert_token_end_index] = [self.schema.role2id[
                                                                                          e[
                                                                                              'event_type'] + '-' +
                                                                                          a[
                                                                                              'role']] * 2 + 2] * (
                                                                                             bert_token_end_index - bert_token_start_index - 1)
                event.arguments.append(argument)
            event.text_seq_tags_id = text_seq_tag_id[1:]
            input_example.events.append(event)
        return input_example

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

    def multi_event_seq_tag_collate_fn(self, datas):
        """
        train phrase: 序列标注：对文本中的论元进行序列标注
        datas[0]: token_ids
        data[1]: seq_tags
        data[2];text
        data[3]:arugments
        data[4]:tokenized_text
        data[5]:seg tag
        data[6]:text_map_seg_idx
        data[7]:seg_idx_map_bert_idx
        data[8]: bert_idx_map_seg_idx
        data[9]:seg_idx_map_text
        data[10]:adj_matrix_subject
        data[11]:adj_matrix_object
        data[12]:adj_matrix_other
        data[13]:kernal_verb
        data[14]:event_id
        data.text_token_id, seq_tag, text, arguments, tokenized_text, data.text_seg_tag,data.text_map_seg_idx,
        data.seg_idx_map_bert_idx,data.bert_idx_map_seg_idx,data.seg_idx_map_text,data.adj_matrix_subject,
        data.adj_matrix_object,data.adj_matrix_others,data.kernal_verb
        """
        seq_lens = []  # 记录每个句子的长度
        text_ids = []  # 训练数据text_token_id
        seq_tags = []  # 记录该batch下文本所对应的论元的序列标注
        raw_texts = []
        texts = []
        arguments = []
        masks_bert = []
        masks_crf = []
        seg_feature = []
        text_map_seg_idxs = []
        seg_idx_map_bert_idxs = []
        bert_idx_map_seg_idxs = []
        seg_idx_map_texts = []
        adj_matrixe_subjects = []
        adj_matrixe_objects = []
        adj_matrixe_others = []
        kernal_verbs = []
        event_labels = []

        max_seq_len = len(max(datas, key=lambda x: len(x[0]))[0])  # 该batch中句子的最大长度
        for data in datas:
            event_labels.append(data[14])
            raw_texts.append(data[2])
            seq_len = len(data[0])
            seq_lens.append(seq_len)
            mask_bert = [1] * seq_len
            mask_crf = [1] * (seq_len - 1)

            texts.append(data[4][1:])
            arguments.append(data[3])

            text_ids.append(data[0] + [self.tokenizer.pad_token_id] * (max_seq_len - seq_len))  # 文本id
            masks_bert.append(mask_bert + [0] * (max_seq_len - seq_len))

            seq_tags.append(data[1] + [0] * (max_seq_len - seq_len))
            masks_crf.append(mask_crf + [0] * (max_seq_len - seq_len))
            kernal_verbs.append(data[13] + [0] * (max_seq_len - seq_len))
            # subject
            trans_adj_matrix_subject = np.zeros((max_seq_len, max_seq_len))
            trans_adj_matrix_subject[:seq_len, :seq_len] = data[10]
            adj_matrixe_subjects.append(trans_adj_matrix_subject)

            # object
            trans_adj_matrix_object = np.zeros((max_seq_len, max_seq_len))
            trans_adj_matrix_object[:seq_len, :seq_len] = data[11]
            adj_matrixe_objects.append(trans_adj_matrix_object)

            # other
            trans_adj_matrix_other = np.zeros((max_seq_len, max_seq_len))
            trans_adj_matrix_other[:seq_len, :seq_len] = data[12]
            adj_matrixe_others.append(trans_adj_matrix_other)

            seg_feature.append(self.encoder_sentence_seg_tag(data[5], max_seq_len))
            text_map_seg_idxs.append(data[6])
            seg_idx_map_bert_idxs.append(data[7])
            bert_idx_map_seg_idxs.append(data[8])
            seg_idx_map_texts.append(data[9])

        # text_ids = torch.LongTensor(np.array(text_ids)).to(self.device)
        # seq_tags = torch.LongTensor(np.array(seq_tags)).to(self.device)
        # masks_bert = torch.ByteTensor(np.array(masks_bert)).to(self.device)
        # masks_crf = torch.ByteTensor(np.array(masks_crf)).to(self.device)
        # seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        # seg_feature = torch.LongTensor(np.array(seg_feature)).to(self.device)
        # kernal_verbs = torch.LongTensor(np.array(kernal_verbs)).to(self.device)
        # adj_matrixes = torch.FloatTensor(np.array(adj_matrixes)).to(self.device)

        text_ids = torch.LongTensor(np.array(text_ids)).cuda()
        seq_tags = torch.LongTensor(np.array(seq_tags)).cuda()
        masks_bert = torch.ByteTensor(np.array(masks_bert)).cuda()
        masks_crf = torch.ByteTensor(np.array(masks_crf)).cuda()
        seq_lens = torch.LongTensor(np.array(seq_lens)).cuda()
        seg_feature = torch.LongTensor(np.array(seg_feature)).cuda()
        kernal_verbs = torch.LongTensor(np.array(kernal_verbs)).cuda()
        adj_matrixes_subjects = torch.FloatTensor(np.array(adj_matrixe_subjects)).cuda()
        adj_matrixe_objects = torch.FloatTensor(np.array(adj_matrixe_objects)).cuda()
        adj_matrixe_others = torch.FloatTensor(np.array(adj_matrixe_others)).cuda()
        event_labels = torch.LongTensor(np.array(event_labels)).cuda()

        return text_ids, seq_lens, masks_bert, masks_crf, texts, arguments, seg_feature, seq_tags, text_map_seg_idxs, \
               seg_idx_map_bert_idxs, bert_idx_map_seg_idxs, seg_idx_map_texts, kernal_verbs, \
               adj_matrixes_subjects, adj_matrixe_objects, adj_matrixe_others,event_labels, raw_texts

    def encoder_sentence_seg_tag(self, sentence_seg_tag, max_sentence_len):
        encoder_matrix = []

        encoder_matrix.extend([self.seg_tag_dict[c] for c in sentence_seg_tag])
        encoder_matrix.extend([self.seg_tag_dict['MASK']] * (max_sentence_len - len(sentence_seg_tag) - 1))
        return encoder_matrix

    def encoder_sentence_pos_tag(self, sentence_pos_tag, max_sentence_len):
        encoder_matrix = []
        encoder_matrix.extend([self.pos_tag_dict[c] for c in sentence_pos_tag])
        encoder_matrix.extend([self.pos_tag_dict['MASK']] * (max_sentence_len - len(sentence_pos_tag) - 1))
        return encoder_matrix

    def encoder_sentence_ner_tag(self, sentence_ner_tag, max_sentence_len):
        encoder_matrix = []

        encoder_matrix.extend([self.ner_tag_dict[c] for c in sentence_ner_tag])
        encoder_matrix.extend([self.ner_tag_dict['MASK']] * (max_sentence_len - len(sentence_ner_tag) - 1))
        return encoder_matrix

    def inference_collate_fn(self, datas):
        """
        test phrase: 序列标注：对文本中的论元进行序列标注
        data 为input_example 对象
        """
        ids = []
        seq_lens = []  # 记录每个句子的长度
        text_ids = []  # 训练数据text_token_id
        texts = []  # 分词之后的文本
        masks_bert = []  # 统一长度，文本内容为1，非文本内容为0
        masks_crf = []
        seg_feature = []
        raw_texts = []
        text_map_seg_idxs = []
        seg_idx_map_bert_idxs = []
        bert_idx_map_seg_idxs = []
        seg_idx_map_texts = []
        adj_matrixe_subjects = []
        adj_matrixe_objects = []
        adj_matrixe_others = []
        kernal_verbs = []
        max_seq_len = len(max(datas, key=lambda x: len(x.text_token_id)).text_token_id)  # 该batch中句子的最大长度
        for data in datas:
            ids.append(data.id)
            text_map_seg_idxs.append(data.text_map_seg_idx)
            seg_idx_map_bert_idxs.append(data.seg_idx_map_bert_idx)
            bert_idx_map_seg_idxs.append(data.bert_idx_map_seg_idx)
            raw_texts.append(data.text)
            seq_len = len(data.text_token_id)  # 句子长度
            seq_lens.append(seq_len)
            mask_bert = [1] * seq_len
            mask_crf = [1] * (seq_len - 1)
            kernal_verbs.append(data.kernal_verb + [0] * (max_seq_len - seq_len))

            # subject
            trans_adj_matrix_subject = np.zeros((max_seq_len, max_seq_len))
            trans_adj_matrix_subject[:seq_len, :seq_len] = data.adj_matrix_subject
            adj_matrixe_subjects.append(trans_adj_matrix_subject)
            # object
            trans_adj_matrix_object = np.zeros((max_seq_len, max_seq_len))
            trans_adj_matrix_object[:seq_len, :seq_len] = data.adj_matrix_object
            adj_matrixe_objects.append(trans_adj_matrix_object)
            # others
            trans_adj_matrix_other = np.zeros((max_seq_len, max_seq_len))
            trans_adj_matrix_other[:seq_len, :seq_len] = data.adj_matrix_others
            adj_matrixe_others.append(trans_adj_matrix_other)

            texts.append(data.text_token[1:])
            text_ids.append(data.text_token_id + [self.tokenizer.pad_token_id] * (max_seq_len - seq_len))  # 文本id
            masks_bert.append(mask_bert + [0] * (max_seq_len - seq_len))
            masks_crf.append(mask_crf + [0] * (max_seq_len - seq_len))
            seg_feature.append(self.encoder_sentence_seg_tag(data.text_seg_tag, max_seq_len))
            seg_idx_map_texts.append(data.seg_idx_map_text)

        text_ids = torch.LongTensor(np.array(text_ids)).cuda()
        masks_bert = torch.ByteTensor(np.array(masks_bert)).cuda()
        masks_crf = torch.ByteTensor(np.array(masks_crf)).cuda()
        seq_lens = torch.LongTensor(np.array(seq_lens)).cuda()
        seg_feature = torch.LongTensor(np.array(seg_feature)).cuda()
        kernal_verbs = torch.LongTensor(np.array(kernal_verbs)).cuda()
        adj_matrixe_subjects = torch.FloatTensor(np.array(adj_matrixe_subjects)).cuda()
        adj_matrixe_objects = torch.FloatTensor(np.array(adj_matrixe_objects)).cuda()
        adj_matrixe_others = torch.FloatTensor(np.array(adj_matrixe_others)).cuda()
        return ids, text_ids, seq_lens, masks_bert, masks_crf, texts, seg_feature, text_map_seg_idxs, \
               seg_idx_map_bert_idxs, bert_idx_map_seg_idxs, seg_idx_map_texts, kernal_verbs, \
               adj_matrixe_subjects, adj_matrixe_objects, adj_matrixe_others, raw_texts
