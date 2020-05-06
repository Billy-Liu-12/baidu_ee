# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 9:22 上午
# @Author  : lizhen
# @FileName: test.py
# @Description:
import argparse
import torch
from tqdm import tqdm
import data_process.data_process as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import utils
from model import torch_crf as module_arch_crf
from torch.utils import data as module_dataloader
from utils.utils import extract_arguments, bert_extract_arguments, inference_feature_collate_fn, get_restrain_crf
import json
import numpy as np
from torch import nn
import pylcs
import pickle
import torch.nn.functional as F
from model.metric import precision, recall, f1


def restrain(start, trans):
    start_np = start.cpu().data.numpy()  # tags
    trans_np = trans.cpu().data.numpy()  # tags tags

    # start 里面 所有的I不能作为第一个，所以赋值-100000
    # trans 里面 O不能接所有的i 所以 【0,2i+2】全部为 -100000
    #            B只能接 B或者自己的i或者o 所以所有的【2i+1,2j+2】为-10000 i不等于j

    for i in range(0, start_np.shape[0]):
        if i > 0 and i % 2 == 0:
            start_np[i] = -10000.0
        if i == 0:
            for j in range(0, trans_np.shape[1]):
                if j > 0 and j % 2 == 0:
                    trans_np[i][j] = -10000.0
        if i % 2 == 1:
            for j in range(0, trans_np.shape[1]):
                if j > 0 and j % 2 == 0 and not j == (i + 1):
                    trans_np[i][j] = -10000.0

    start = nn.Parameter(torch.FloatTensor(start_np)).cuda()
    trans = nn.Parameter(torch.FloatTensor(trans_np)).cuda()
    return start, trans


def inference(config):
    logger = config.get_logger('test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    word_embedding = config.init_ftn('word_embedding', utils)
    # setup dataset, data_loader instances
    test_set_ = config.init_obj('test1_set', module_data, device=device)
    with open('data/bert_feature/test_feature.pkl', 'rb') as f:
        test_set = pickle.load(f)
    test_dataloader = config.init_obj('data_loader', data_loader, test_set, collate_fn=inference_feature_collate_fn)

    # build model architecture
    model = config.init_obj('model_arch', module_arch, num_classes=435)
    logger.info(model)
    crf_model = config.init_obj('model_arch_crf', module_arch_crf, 435)
    logger.info(crf_model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    crf_state_dict = checkpoint['crf_state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
        crf_model = torch.nn.DataParallel(crf_model)
    model.load_state_dict(state_dict)
    crf_model.load_state_dict(crf_state_dict)

    crf_model.start_transitions, crf_model.transitions = \
        restrain(crf_model.start_transitions, crf_model.transitions)

    # prepare model for testing
    model = model.to(device)
    crf_model = crf_model.to(device)

    # O:0 B：2id+1 I：2id+2
    '''
        self.start_transitions = nn.Parameter(torch.empty(num_tags)).cuda()
        self.end_transitions = nn.Parameter(torch.empty(num_tags)).cuda()
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags)).cuda()
    '''

    model.eval()
    crf_model.eval()

    # inference
    f_txt = open('result.txt', 'w', encoding='utf8')
    f_result = open('result.json', 'w', encoding='utf8')
    schema = test_set_.schema
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_dataloader)):

            ids, sentence_feature, text_ids, seq_lens, masks_bert, masks_crf, texts = batch_data

            out_class, out_event, out_tag = model(sentence_feature, seq_lens)

            best_path = crf_model.decode(emissions=out_tag, mask=masks_crf)
            for id, text, path in zip(ids, texts, best_path):
                arguments = bert_extract_arguments(text, pred_tag=path, schema=schema)
                event_list = []
                for k, v in arguments.items():
                    event_list.append({
                        'event_type': v[0],
                        'arguments': [{
                            'role': v[1],
                            'argument': k
                        }]
                    })
                res = {}
                res['id'] = id
                res['event_list'] = event_list
                l = json.dumps(res, ensure_ascii=False)
                f_result.write(l + '\n')
                res = {}
                res['id'] = id
                res['text'] = text
                res['event_list'] = event_list
                f_txt.write(str(res) + '\n')
    f_result.close()
    f_txt.close()


def bert_evalution(config):
    def evaluate(batch_pred_tag, batch_text, batch_arguments):
        """评测函数（跟官方评测结果不一定相同，但很接近）
        """

        X, Y, Z = 1e-10, 1e-10, 1e-10

        for pred_tag, text, arguments in zip(batch_pred_tag, batch_text, batch_arguments):

            inv_arguments_label = {v: k for k, v in arguments.items()}
            pred_arguments = bert_extract_arguments(text, pred_tag, schema)
            pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
            Y += len(pred_inv_arguments)
            Z += len(inv_arguments_label)
            for k, v in pred_inv_arguments.items():
                if k in inv_arguments_label:
                    # 用最长公共子串作为匹配程度度量
                    l = pylcs.lcs(v, inv_arguments_label[k])
                    X += 2. * l / (len(v) + len(inv_arguments_label[k]))
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        return X, Y, Z

    logger = config.get_logger('test')
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # setup dataset, data_loader instances
    data_set = config.init_obj('data_set', module_data, device=device)
    valid_dataloader = config.init_obj('data_loader', module_dataloader, data_set.valid_set,
                                       collate_fn=data_set.seq_tag_collate_fn)
    # build model architecture
    model = config.init_obj('model_arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    # prepare model for testing
    model = model.to(device)
    # model.crf.to(device)

    model.eval()
    model.crf.eval()
    # model.crf.reset_parameters()

    # inference
    schema = data_set.schema
    recalls = []
    precisions = []
    f1s = []
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(valid_dataloader)):
            text_ids, seq_lens, masks_bert, masks_crf, texts, arguments, class_label, event_label, seq_tags = batch_data
            pred_tags = model(text_ids, seq_lens, masks_bert)
            # max_prob, best_path = torch.max(F.softmax(pred_tags, dim=2), dim=2)
            pred_tags = torch.log_softmax(pred_tags, dim=2)
            best_path = model.crf.decode(pred_tags, masks_crf)
            X, Y, Z = evaluate(best_path, texts, arguments)
            recalls.append(recall(X, Y, Z))
            precisions.append(precision(X, Y, Z))
            f1s.append(f1(X, Y, Z))
        print('precision:{},recall:{},f1:{}'.format(sum(precisions) / len(precisions), sum(recalls) / len(recalls),
                                                    sum(f1s) / len(f1s)))


def bert_inference(config):
    logger = config.get_logger('test')
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # setup dataset, data_loader instances
    data_set = config.init_obj('test1_set', module_data, device=device)
    test_dataloader = config.init_obj('data_loader', module_dataloader, data_set,
                                      collate_fn=data_set.inference_collate_fn)

    # build model architecture
    restrain = get_restrain_crf(device)
    model = config.init_obj('model_arch', module_arch, seg_vocab_size=5, pos_vocab_size=32, ner_vocab_size=5,
                            restrain=restrain, device=device)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        device_ids = list(map(lambda x:int(x),config.config['device_id'].split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # inference
    f_txt = open('result.txt', 'w', encoding='utf8')
    f_result = open('result.json', 'w', encoding='utf8')
    schema = data_set.schema
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_dataloader)):
            ids, text_ids, seq_lens, masks_bert, masks_crf, texts,seg_feature,pos_feature,ner_feature = batch_data
            pred_class, pred_event, pred_tags = model(text_ids, seq_lens, masks_bert, seg_feature, pos_feature,ner_feature)
            best_path = model.module.crf.decode(pred_tags, masks_crf)
            for id, text, path in zip(ids, texts, best_path):
                arguments = bert_extract_arguments(text, pred_tag=path, schema=schema)
                event_list = []
                for k, v in arguments.items():
                    event_list.append({
                        'event_type': v[0],
                        'arguments': [{
                            'role': v[1],
                            'argument': k
                        }]
                    })
                res = {}
                res['id'] = id

                res['event_list'] = event_list
                l = json.dumps(res, ensure_ascii=False)
                f_result.write(l + '\n')
                res['text'] = ''.join(text)
                res['event_list'] = event_list
                l = json.dumps(res, ensure_ascii=False)
                f_txt.write(l + '\n')
    f_result.close()
    f_txt.close()


def main(config_file):
    args = argparse.ArgumentParser(description='event extraction')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='saved/models/seq_label/0429_083739/model_best.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='2,1,0,3', type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    # 是否使用bert作为预训练模型
    if 'bert' in config.config['config_file_name'].lower():
        bert_inference(config)
        # bert_evalution(config)
    else:
        inference(config)






if __name__ == '__main__':
    main('configs/roberta_crf_0.json')
