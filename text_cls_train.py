# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 9:28 下午
# @Author  : lizhen
# @FileName: text_cls_train.py
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
from trainer.text_cls_trainer import TextCLSTrainer
import transformers as optimization
from torch.utils import data as module_dataloader


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def text_cls_train(config):
    logger = config.get_logger('train')
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # setup data_set instances
    data_set = config.init_obj('data_set', module_dataset,device=device)
    # setup data_loader instances
    train_dataloader = config.init_obj('data_loader', module_dataloader, data_set.train_set,
                                       collate_fn=data_set.multi_event_seq_tag_collate_fn)
    valid_dataloader = config.init_obj('data_loader', module_dataloader, data_set.valid_set,
                                       collate_fn=data_set.multi_event_seq_tag_collate_fn)
    # train_dataloader = valid_dataloader

    # build model architecture, then print to console
    model = config.init_obj('model_arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = [getattr(module_loss, crit) for crit in config['loss']]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    bert_params = filter(lambda p: p.requires_grad, model.bert.parameters())
    attention_cls_params = filter(lambda p:p.requires_grad,model.att_class_classifier.parameters())
    # fc_params = filter(lambda p: p.requires_grad, model.fc.parameters())

    optimizer = config.init_obj('optimizer', optimization, [
                                                            {'params': bert_params, 'lr': 2e-5, "weight_decay": 0.0},
                                                            {'params': attention_cls_params, 'lr': 1e-4, "weight_decay": 0.0},
                                                            # {'params': fc_tags_params, 'lr': 7e-4, "weight_decay": 0.0}
                                ])



    lr_scheduler = config.init_obj('lr_scheduler', optimization.optimization, optimizer,
                                   num_training_steps=int(len(train_dataloader.dataset) / train_dataloader.batch_size))

    trainer = TextCLSTrainer(model, criterion, metrics, optimizer,
                          config=config,
                          train_iter=train_dataloader,
                          valid_iter=valid_dataloader,
                          schema=data_set.schema,
                          device=device,
                          lr_scheduler=lr_scheduler)

    trainer.train()


def run_main(config_file):
    args = argparse.ArgumentParser(description='text classification')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default='0,1,2,3', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    text_cls_train(config)

if __name__ == '__main__':
    run_main('configs/text_cls.json')