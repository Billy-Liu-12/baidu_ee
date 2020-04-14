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

    # prepare model for testing
    model = model.to(device)
    crf_model = crf_model.to(device)
    model.eval()
    crf_model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_dataloader)):
            text_ids, seq_lens, masks, texts = batch_data
            output = model(text_ids, seq_lens).squeeze()

            # save inference result, or do something with output here


            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            # batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='event extraction')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='saved/models/seqlabel/0414_162250/model_best.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='2', type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
