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
# from utils.pytorch_pretrained import optimization
from torch import optim as optimization
# import transformers as optimization
from model import torch_crf as module_arch_crf
from torch.utils import data as data_loader
from utils import utils
from utils.utils import bert_extract_arguments,Feature_example,feature_collate_fn,Test_example
from tqdm import tqdm
import pickle



# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)



def get_bert_feature(config):
    logger = config.get_logger('test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup data_set, data_loader instances
    data_set = config.init_obj('data_set', module_data, device=device)
    # setup data_loader instances
    train_dataloader = config.init_obj('data_loader', data_loader, data_set.train_set, collate_fn=data_set.collate_fn)
    valid_dataloader = config.init_obj('data_loader', data_loader, data_set.valid_set, collate_fn=data_set.collate_fn)
    # setup dataset, data_loader instances
    test_set = config.init_obj('test1_set', module_data,device=device)
    test_dataloader = config.init_obj('data_loader', data_loader, test_set, collate_fn=test_set.inference_collate_fn)
    # train_dataloader = valid_dataloader


    # build model architecture
    model = config.init_obj('model_arch', module_arch, num_classes=data_set.num_tag_labels)
    logger.info(model)
    crf_model = config.init_obj('model_arch_crf', module_arch_crf, data_set.num_tag_labels)
    logger.info(crf_model)


    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)


    # prepare model for testing
    model = model.to(device)
    model.eval()

    # get bert feature
    with torch.no_grad():
        train_examples = []
        for i, batch_data in enumerate(tqdm(train_dataloader)):
            text_ids, seq_lens, masks_bert, masks_crf, texts, arguments,class_labels,event_labels, seq_tags = batch_data
            output,sentence_feature = model(text_ids, seq_lens, masks_bert)
            sentence_feature = sentence_feature.cpu().numpy()
            seq_lens = seq_lens.cpu().numpy()
            masks_bert = masks_bert.cpu().numpy()
            masks_crf = masks_crf.cpu().numpy()
            seq_tags = seq_tags.cpu().numpy()
            for e in zip(sentence_feature,text_ids, seq_lens, masks_bert, masks_crf, texts, arguments,class_labels,event_labels, seq_tags):
                
                example = Feature_example(*e)
                train_examples.append(example)

        valid_examples = []
        for i, batch_data in enumerate(tqdm(valid_dataloader)):
            text_ids, seq_lens, masks_bert, masks_crf, texts, arguments, class_labels, event_labels, seq_tags = batch_data
            output, sentence_feature = model(text_ids, seq_lens, masks_bert)
            sentence_feature = sentence_feature.cpu().numpy()
            seq_lens = seq_lens.cpu().numpy()
            masks_bert = masks_bert.cpu().numpy()
            masks_crf = masks_crf.cpu().numpy()
            seq_tags = seq_tags.cpu().numpy()
            for e in zip(sentence_feature,text_ids, seq_lens, masks_bert, masks_crf, texts, arguments,class_labels,event_labels, seq_tags):
                example = Feature_example(*e)
                valid_examples.append(example)

        test_examples = []
        for i, batch_data in enumerate(tqdm(test_dataloader)):
            ids, text_ids, seq_lens, masks_bert, masks_crf, texts = batch_data

            output, sentence_feature = model(text_ids, seq_lens, masks_bert)

            sentence_feature = sentence_feature.cpu().numpy()
            seq_lens = seq_lens.cpu().numpy()
            masks_bert = masks_bert.cpu().numpy()
            masks_crf = masks_crf.cpu().numpy()
            for t_e in zip(ids,sentence_feature,text_ids,seq_lens,masks_bert,masks_crf,texts):
                text_example =Test_example(*t_e)
                test_examples.append(text_example)


    with open('data/bert_feature/train_feature.pkl','wb') as f:
        pickle.dump(train_examples,f)
    with open('data/bert_feature/valid_feature.pkl','wb') as f:
        pickle.dump(valid_examples,f)
    with open('data/bert_feature/test1_feature.pkl','wb') as f:
        pickle.dump(test_examples,f)
    print('success!')




def bert_train(config):
    logger = config.get_logger('train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    no_decay = ["bias", "LayerNorm.weight"]

    # setup data_set, data_loader instances
    data_set = config.init_obj('data_set', module_data, device=device)
    with open('data/bert_feature/train_feature.pkl','rb') as f:
        train_set = pickle.load(f)
    with open('data/bert_feature/valid_feature.pkl','rb') as f:
        valid_set = pickle.load(f)



    # setup data_loader instances
    # train_dataloader = config.init_obj('data_loader', data_loader, data_set.train_set, collate_fn=data_set.seq_tag_collate_fn)
    # valid_dataloader = config.init_obj('data_loader', data_loader, data_set.valid_set, collate_fn=data_set.seq_tag_collate_fn)

    train_dataloader = config.init_obj('data_loader', data_loader, train_set,collate_fn=feature_collate_fn)
    valid_dataloader = config.init_obj('data_loader', data_loader, valid_set,collate_fn=feature_collate_fn)
    # train_dataloader = valid_dataloader

    # build model architecture, then print to console
    model = config.init_obj('model_arch', module_arch, num_classes=data_set.num_tag_labels)
    logger.info(model)
    crf_model = config.init_obj('model_arch_crf', module_arch_crf, data_set.num_tag_labels)
    logger.info(crf_model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # bert_params = [p for n, p in model.bert.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)]
    bert_params = []
    crf_params = [p for n, p in crf_model.named_parameters() if not any(nd in n for nd in no_decay)]

    # if 'rnn' not in config.config['config_file_name']:
    #     params = [p for n, p in model.fc.named_parameters() if not any(nd in n for nd in no_decay)]
    # else:
    #     params = [p for n, p in model.fc.named_parameters() if not any(nd in n for nd in no_decay)]+[p  for n, p in model.rnn.named_parameters() if not any(nd in n for nd in no_decay)]
    params = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]


    if bert_params:
        optimizer = config.init_obj('optimizer', optimization, [{'params':bert_params,'lr':1e-5,"weight_decay": 0.0},
                                                                {'params':params,'lr':1e-5,"weight_decay": 0.0},
                                                                {'params':crf_params,'lr':1e-3,"weight_decay": 0.0}])
    else:
        # optimizer = config.init_obj('optimizer', optimization, [{'params':params,'lr':1e-3,"weight_decay": 0.0},
        #                                                         {'params':crf_params,'lr':1e-2,"weight_decay": 0.0}])
        optimizer = config.init_obj('optimizer', optimization, [{'params':params},
                                                                {'params':crf_params,'lr':1e-2,"weight_decay": 0.0}])

    lr_scheduler = config.init_obj('lr_scheduler', optimization.optimization, optimizer,num_training_steps=int(len(train_dataloader.dataset)/train_dataloader.batch_size))

    # lr_scheduler = config.init_obj('lr_scheduler', optimization.lr_scheduler, optimizer)

    trainer = BertTrainer(model, crf_model,criterion, metrics, optimizer,
                          config=config,
                          train_iter=train_dataloader,
                          valid_iter=valid_dataloader,
                          schema=data_set.schema,
                          lr_scheduler=lr_scheduler)


    trainer.train()


def run_main(config_file):
    args = argparse.ArgumentParser(description='text classification')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default='2', type=str,
                      help='indices of GPUs to enable (default: all)')
    # /data/lz/workspace/EE/saved/best/models/0417_235821/model_best.pth
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)


    # get_bert_feature(config)
    bert_train(config)



def pipeline():
    run_main('configs/roberta_crf.json')


if __name__ == '__main__':

    pipeline()
