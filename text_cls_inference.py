# -*- coding: utf-8 -*-
# @Time    : 2020/5/3 8:27 上午
# @Author  : lizhen
# @FileName: text_cls_inference.py
# @Description:
import argparse
import torch
from tqdm import tqdm
import data_process.data_process as module_data
import model.model as module_arch
from parse_config import ConfigParser
from torch.utils import data as module_dataloader
import numpy as np
from collections import defaultdict
import os
import json


def inference(config):
    logger = config.get_logger('test')
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # setup dataset, data_loader instances
    data_set = config.init_obj('test1_set', module_data, device=device)
    test_dataloader = config.init_obj('data_loader', module_dataloader, data_set,
                                      collate_fn=data_set.inference_collate_fn)

    # build model architecture
    model = config.init_obj('model_arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        device_ids = list(map(lambda x: int(x), config.config['device_id'].split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # inference
    res = defaultdict(list)
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_dataloader)):
            ids, text_ids, seq_lens, masks_bert, texts_token,texts = batch_data
            pred_cls = model(text_ids, seq_lens, masks_bert)
            pred_cls = pred_cls.detach().cpu().numpy()
            pred_cls = np.where(pred_cls > 0.5, 1, 0)

            for id_,text, cls in zip(ids,texts, pred_cls):
                cls_col = [col for col, k in enumerate(cls) if k == 1]
                if not cls_col:  # 如果没有召回
                    res[9].append((id_,text))
                for key in cls_col:
                    res[key].append((id_,text))

        for key,id_text in res.items():
            with open(os.path.join('data/cls_result','test1_{}.json'.format(key)),'w',encoding='utf8') as f:
                for id_,text in id_text:
                    f.write(json.dumps({'id':id_,'text':text},ensure_ascii=False)+'\n')





def main(config_file):
    args = argparse.ArgumentParser(description='event extraction')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='saved/text_cls/models/seq_label/0501_131931/model_best.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='2,1,0,3', type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    inference(config)


if __name__ == '__main__':
    main('configs/text_cls.json')
