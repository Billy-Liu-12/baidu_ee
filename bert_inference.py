# -*- coding: utf-8 -*-
# @Time    : 2020/5/3 11:48 上午
# @Author  : lizhen
# @FileName: bert_inference.py
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


def bert_inference(config, class_id):
    logger = config.get_logger('test')
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # setup dataset, data_loader instances
    data_set = config.init_obj('test1_set', module_data, device=device)
    test_dataloader = config.init_obj('data_loader', module_dataloader, data_set,
                                      collate_fn=data_set.inference_collate_fn)

    # build model architecture
    restrain = get_restrain_crf(device, data_set.schema.class_tag_num[class_id])
    model = config.init_obj('model_arch', module_arch, num_tags=data_set.schema.class_tag_num[class_id],event_label_num=data_set.schema.event_num[class_id],
                            restrain=restrain)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        device_ids = list(map(lambda x: int(x), config.config['device_id'].split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.cuda()
    model.eval()

    # inference
    f_result = open('data/tag_result/result_{}.json'.format(class_id), 'w', encoding='utf8')
    schema = data_set.schema
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_dataloader)):
            ids, text_ids, seq_lens, masks_bert, masks_crf, texts, seg_feature, text_map_seg_idxs, seg_idx_map_bert_idxs, bert_idx_map_seg_idxs, seg_idx_map_texts,kernal_verbs,adj_matrixe_subjects,adj_matrixe_objects,adj_matrixe_others,   raw_texts = batch_data
            pred_tags,_ = model(text_ids, seq_lens, torch.max(seq_lens).item(), masks_bert, seg_feature,kernal_verbs,adj_matrixe_subjects,adj_matrixe_objects,adj_matrixe_others, )
            # best_path = model.module.crf.decode(pred_tags, masks_crf)
            score,best_path = model.crf.decode(pred_tags, masks_crf)
            for id, text, path, text_map_seg_idx, seg_idx_map_bert_idx, bert_idx_map_seg_idx, seg_idx_map_text, raw_text in zip(
                    ids, texts, best_path, text_map_seg_idxs, seg_idx_map_bert_idxs, bert_idx_map_seg_idxs,
                    seg_idx_map_texts, raw_texts):
                # text, pred_tag, schema, class_id,text_map_seg_idx,seg_idx_map_bert_idx,bert_idx_map_seg_idx,raw_text
                arguments = bert_extract_arguments(text, pred_tag=path, schema=schema, class_id=class_id,
                                                   text_map_seg_idx=text_map_seg_idx,
                                                   seg_idx_map_bert_idx=seg_idx_map_bert_idx,
                                                   bert_idx_map_seg_idx=bert_idx_map_seg_idx,seg_idx_map_text=seg_idx_map_text, raw_text=raw_text)
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
                res['text'] = raw_text
                res['event_list'] = event_list
                l = json.dumps(res, ensure_ascii=False)
                f_result.write(l + '\n')
    f_result.close()


def main(config_file, class_id, resume_path):
    args = argparse.ArgumentParser(description='event extraction')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=resume_path, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='2', type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    # 是否使用bert作为预训练模型

    bert_inference(config, class_id)


if __name__ == '__main__':
    main('configs/roberta_crf_0.json', 0, 'saved/tag_/0/models/seq_label/0509_163927/model_best.pth')
    main('configs/roberta_crf_1.json', 1, 'saved/tag_/1/models/seq_label/0509_170708/model_best.pth')
    main('configs/roberta_crf_2.json', 2, 'saved/tag_/2/models/seq_label/0509_142222/model_best.pth')
    main('configs/roberta_crf_3.json', 3, 'saved/tag_/3/models/seq_label/0509_211412/model_best.pth')
    main('configs/roberta_crf_4.json', 4, 'saved/tag_/4/models/seq_label/0509_223255/model_best.pth')
    main('configs/roberta_crf_5.json', 5, 'saved/tag_/5/models/seq_label/0509_143850/model_best.pth')
    main('configs/roberta_crf_6.json', 6, 'saved/tag_/6/models/seq_label/0509_190611/model_best.pth')
    main('configs/roberta_crf_7.json', 7, 'saved/tag_/7/models/seq_label/0509_195108/model_best.pth')
    main('configs/roberta_crf_8.json', 8, 'saved/tag_/8/models/seq_label/0509_140730/model_best.pth')
