# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 12:46 下午
# @Author  : lizhen
# @FileName: bert_train.py
# @Description:

import argparse
import collections
import torch
import numpy as np
import data_process.data_process as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import BertTrainer
import transformers as optimization
from model import torch_crf as module_arch_crf
from torch.utils import data as module_dataloader
from utils.utils import bert_extract_arguments, Feature_example, feature_collate_fn, Test_example, get_restrain_crf
from tqdm import tqdm
import os

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def roberta_train(config, class_id):
    logger = config.get_logger('train')
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # setup data_set instances
    data_set = config.init_obj('data_set', module_dataset, device=device)
    # setup data_loader instances
    train_dataloader = config.init_obj('data_loader', module_dataloader, data_set.train_set, batch_size=16,
                                       collate_fn=data_set.multi_event_seq_tag_collate_fn)
    valid_dataloader = config.init_obj('data_loader', module_dataloader, data_set.valid_set, batch_size=1,
                                       collate_fn=data_set.multi_event_seq_tag_collate_fn)

    # build model architecture, then print to console
    restrain = get_restrain_crf(device, data_set.schema.class_tag_num[class_id])
    model = config.init_obj('model_arch', module_arch, num_tags=data_set.schema.class_tag_num[class_id]
                            ,event_label_num=data_set.schema.event_num[class_id],
                            restrain=restrain)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = [getattr(module_loss, crit) for crit in config['loss']]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler

    bert_params = filter(lambda p: p.requires_grad, model.bert.parameters())
    # rnn_params = filter(lambda p:p.requires_grad,model.rnn.parameters())
    gcn_params = [*filter(lambda p: p.requires_grad, model.fc_gcn_in.parameters())] + \
                 [*filter(lambda p: p.requires_grad, model.fc_gcn_out.parameters())] + \
                 [*filter(lambda p: p.requires_grad, model.gcn_other_1.parameters())] + \
                 [*filter(lambda p: p.requires_grad, model.gcn_other_2.parameters())] + \
                 [*filter(lambda p: p.requires_grad, model.gcn_other_3.parameters())] + \
                 [*filter(lambda p: p.requires_grad, model.gcn_subject_1.parameters())] + \
                 [*filter(lambda p: p.requires_grad, model.gcn_subject_2.parameters())] + \
                 [*filter(lambda p: p.requires_grad, model.gcn_subject_3.parameters())] + \
                 [*filter(lambda p: p.requires_grad, model.gcn_object_1.parameters())] + \
                 [*filter(lambda p: p.requires_grad, model.gcn_object_2.parameters())] + \
                 [*filter(lambda p: p.requires_grad, model.gcn_object_3.parameters())]

    fc_bert_params = filter(lambda p: p.requires_grad, model.fc_bert.parameters())
    fc_event_params = filter(lambda p:p.requires_grad,model.fc_event_label.parameters())
    fc_tags_params = filter(lambda p: p.requires_grad, model.fc_tags.parameters())
    crf_params = filter(lambda p: p.requires_grad, model.crf.parameters())

    optimizer = config.init_obj('optimizer', optimization, [
        {'params': bert_params, 'lr': 2e-5, "weight_decay": 0.0001},
        {'params': gcn_params, 'lr': 6e-4, "weight_decay": 0.00001},
        {'params': fc_event_params, 'lr': 7e-4, "weight_decay": 0.00001},
        {'params': fc_tags_params, 'lr': 7e-4, "weight_decay": 0.00001},
        {'params': fc_bert_params, 'lr': 7e-4, "weight_decay": 0.00001},
        {'params': crf_params, 'lr': 7e-4, "weight_decay": 0.00001}])

    lr_scheduler = config.init_obj('lr_scheduler', optimization.optimization, optimizer,
                                   num_training_steps=int(len(train_dataloader.dataset) / train_dataloader.batch_size))

    trainer = BertTrainer(model, criterion, metrics, optimizer,
                          config=config,
                          train_iter=train_dataloader,
                          valid_iter=valid_dataloader,
                          schema=data_set.schema,
                          class_id=class_id,
                          device=device,
                          lr_scheduler=lr_scheduler)

    save_dir = trainer.train()
    return save_dir


def run_main(config_file, class_id, model_dir):
    args = argparse.ArgumentParser(description='sequence tagging')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default='3', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=model_dir, type=str,
                      help='path to latest checkpoint (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    save_dir = roberta_train(config, class_id)
    return save_dir


if __name__ == '__main__':
    run_main('configs/roberta_crf_8.json', 8, None)
    run_main('configs/roberta_crf_2.json', 2, None)
    run_main('configs/roberta_crf_5.json', 5, None)
    run_main('configs/roberta_crf_0.json', 0, None)
    run_main('configs/roberta_crf_1.json', 1, None)
    run_main('configs/roberta_crf_6.json', 6, None)
    run_main('configs/roberta_crf_7.json', 7, None)
    run_main('configs/roberta_crf_3.json', 3, None)
    run_main('configs/roberta_crf_4.json', 4, None)
