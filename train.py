import argparse
import collections
import torch
import numpy as np
import data_process.data_process as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer, BertTrainer
from utils.pytorch_pretrained import optimization
from model import torch_crf as module_arch_crf
from torch.utils import data as data_loader
from utils import utils

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    word_embedding = config.init_ftn('word_embedding', utils)
    # setup data_set instances
    train_set = config.init_obj('train_set', module_data, word_embedding=word_embedding, device=device)
    valid_set = config.init_obj('valid_set', module_data, word_embedding=word_embedding, device=device)

    # setup data_loader instances
    train_dataloader = config.init_obj('data_loader', data_loader, train_set, collate_fn=train_set.seq_tag_collate_fn)
    valid_dataloader = config.init_obj('data_loader', data_loader, valid_set, collate_fn=valid_set.seq_tag_collate_fn)
    # train_dataloader = valid_dataloader

    # build model architecture, then print to console
    model = config.init_obj('model_arch', module_arch, word_embedding=word_embedding,
                            output_dim=train_set.num_tag_labels)
    # logger.info(model)
    crf_model = config.init_obj('model_arch_crf', module_arch_crf, train_set.num_tag_labels)
    logger.info(crf_model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    # metrics = [getattr(module_metric, met) for met in config['metrics']]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = [*filter(lambda p: p.requires_grad, model.parameters())] + [
        *filter(lambda p: p.requires_grad, crf_model.parameters())]
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, crf_model, criterion, metrics, optimizer,
                      config=config,
                      train_iter=train_dataloader,
                      valid_iter=valid_dataloader,
                      schema=train_set.schema,
                      lr_scheduler=lr_scheduler)

    trainer.train()


def bert_train(config):
    logger = config.get_logger('train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setup data_set, data_loader instances
    train_set = config.init_obj('train_set', module_data, device=device)
    valid_set = config.init_obj('valid_set', module_data, device=device)
    # setup data_loader instances
    train_dataloader = config.init_obj('data_loader', data_loader, train_set, collate_fn=train_set.seq_tag_collate_fn)
    valid_dataloader = config.init_obj('data_loader', data_loader, valid_set, collate_fn=valid_set.seq_tag_collate_fn)
    # train_dataloader = valid_dataloader

    # build model architecture, then print to console
    model = config.init_obj('model_arch', module_arch, num_classes=train_set.num_tag_labels)
    # logger.info(model)
    crf_model = config.init_obj('model_arch_crf', module_arch_crf, train_set.num_tag_labels)
    logger.info(crf_model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = [*filter(lambda p: p.requires_grad, model.parameters())] + [
        *filter(lambda p: p.requires_grad, crf_model.parameters())]

    optimizer = config.init_obj('optimizer', optimization, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = BertTrainer(model, crf_model,criterion, metrics, optimizer,
                          config=config,
                          train_iter=train_dataloader,
                          valid_iter=valid_dataloader,
                          schema=train_set.schema,
                          lr_scheduler=lr_scheduler)

    trainer.train()


def run_main(config_file):
    args = argparse.ArgumentParser(description='text classification')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='2', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    # 是否使用bert作为预训练模型
    if 'bert' in config.config['config_file_name'].lower():
        bert_train(config)
    else:
        main(config)


def pipeline():
    run_main('configs/bert_rnn_crf.json')


if __name__ == '__main__':
    pipeline()
