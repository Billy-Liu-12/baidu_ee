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
from model.torch_crf import CRF
from torch.utils import data as data_loader
from utils.utils import extract_arguments
import json
import numpy as np
from torch import nn

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

    start = nn.Parameter(torch.from_numpy(start_np)).cuda()
    trans = nn.Parameter(torch.from_numpy(trans_np)).cuda()
    return start, trans


def main(config):
    logger = config.get_logger('test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    word_embedding = config.init_ftn('word_embedding', utils)
    # setup dataset, data_loader instances
    test_set = config.init_obj('test1_set', module_data, word_embedding=word_embedding, device=device)
    test_dataloader = config.init_obj('data_loader', data_loader, test_set, collate_fn=test_set.inference_collate_fn)

    # build model architecture
    model = config.init_obj('model_arch', module_arch, word_embedding=word_embedding,
                            output_dim=test_set.num_tag_labels)
    logger.info(model)
    crf_model = CRF(test_set.num_tag_labels, batch_first=True)
    logger.info(crf_model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

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

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # inference
    f_result = open('result.json', 'w', encoding='utf8')
    schema = test_set.schema
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_dataloader)):
            ids, text_ids, seq_lens, masks, texts = batch_data
            output = model(text_ids, seq_lens)

            best_path = crf_model.decode(emissions=output, mask=masks)
            for id, text, path in zip(ids, texts, best_path):
                arguments = extract_arguments(text, pred_tag=path, schema=schema)
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
                # res['text'] = text
                res['event_list'] = event_list
                l = json.dumps(res, ensure_ascii=False)
                f_result.write(l + '\n')
    f_result.close()

    # save inference result, or do something with output here

    # computing loss, metrics on test set
    # loss = loss_fn(output, target)
    # batch_size = data.shape[0]
    # total_loss += loss.item() * batch_size
    # for i, metric in enumerate(metric_fns):
    #     total_metrics[i] += metric(output, target) * batch_size

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='event extraction')
    args.add_argument('-c', '--config', default='configs/lstm_crf.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='saved/models/seq_label/0414_233012/checkpoint-epoch42.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='1', type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)