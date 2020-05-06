# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 5:47 下午
# @Author  : lizhen
# @FileName: bert_trainer.py
# @Description:
import numpy as np
import torch
from base.base_trainer import BaseTrainer
from utils.utils import inf_loop, MetricTracker, extract_arguments, bert_extract_arguments
from time import time
import pylcs
import torch.nn.functional as F
from model.loss import cut_crossentropy_loss
import math
import random
from sklearn.metrics import f1_score, precision_score, recall_score


class BertTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, train_iter, valid_iter, device, schema,
                 class_id,
                 test_iter=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.class_id = class_id
        self.device = device
        self.train_iter, self.valid_iter, self.test_iter = train_iter, valid_iter, test_iter
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_iter)
        else:
            # iteration-based training
            self.data_loader = inf_loop(train_iter)
            self.len_epoch = len_epoch
        self.schema = schema
        self.do_validation = self.valid_iter is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_iter.batch_size))

        # del self.optimizer.param_groups[2]
        self.train_metrics = MetricTracker('total_loss', 'tags_loss', 'crf_loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('total_loss', 'tags_loss', 'crf_loss','P_all','R_all','F_all',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # self.cross_entropy_weight_ = [1.0] * schema.class_tag_num[class_id]
        self.cross_entropy_weight_ = [1.0] * schema.class_tag_num[class_id]
        for i in range(1, schema.class_tag_num[class_id]):
            if i % 2 == 1:
                self.cross_entropy_weight_[i] = 1.5
        self.cross_entropy_weight_[0] = 0.005

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        t1 = time()
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch_data in enumerate(self.train_iter):
            self.optimizer.zero_grad()
            text_ids, seq_lens, masks_bert, masks_crf, texts, arguments, seg_feature, seq_tags, text_map_seg_idxs, \
            seg_idx_map_bert_idxs, bert_idx_map_seg_idxs, seg_idx_map_texts,kernal_verbs,adj_matrixes, raw_texts = batch_data
            pred_tags = self.model(text_ids, seq_lens, torch.max(seq_lens).item(),masks_bert, seg_feature,kernal_verbs,adj_matrixes,)

            # 多卡
            # crf_loss = self.model.module.crf(emissions=pred_tags, mask=masks_crf, tags=seq_tags, reduction='mean')
            # 单卡
            crf_loss = self.model.crf(emissions=pred_tags, mask=masks_crf, tags=seq_tags, reduction='mean')
            # tags_loss = self.criterion[0](pred_tags, seq_tags, self.device)
            tags_loss = cut_crossentropy_loss(pred_tags, seq_tags, self.device, cut_side=0.5,
                                              weight_=self.cross_entropy_weight_)
            loss = tags_loss + crf_loss
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('tags_loss', tags_loss.item())
            self.train_metrics.update('crf_loss', crf_loss.item())
            self.train_metrics.update('total_loss', loss.item())

            # 多卡
            # best_path = self.model.module.crf.decode(emissions=pred_tags, mask=masks_crf)
            # 单卡
            best_path = self.model.crf.decode(emissions=pred_tags, mask=masks_crf)
            X, Y, Z = self.evaluate(best_path, texts, arguments, text_map_seg_idxs, seg_idx_map_bert_idxs,
                                    bert_idx_map_seg_idxs, seg_idx_map_texts, raw_texts)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(X, Y, Z))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

            # torch.cuda.empty_cache()
        log = self.train_metrics.result()

        if self.do_validation:
            log_f = open('data/log_argument/log_classid_{}_epoch_{}.txt'.format(self.class_id,epoch), 'w', encoding='utf8')
            val_log = self._valid_epoch(epoch,log_f)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        print('spending time:', time() - t1)
        return log

    def _valid_epoch(self, epoch,log_f):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.model.eval()
        # self.model.crf.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            X_all = 0.
            Y_all = 0.00000000001
            Z_all = 0.00000000001

            for batch_idx, batch_data in enumerate(self.valid_iter):
                text_ids, seq_lens, masks_bert, masks_crf, texts, arguments, seg_feature, seq_tags, text_map_seg_idxs, \
                seg_idx_map_bert_idxs, bert_idx_map_seg_idxs, seg_idx_map_texts,kernal_verbs,adj_matrixes, raw_texts = batch_data

                # context, seq_len, mask_bert,mask_crf,class_label,event_label,seq_tags
                pred_tags = self.model(text_ids, seq_lens, torch.max(seq_lens).item(), masks_bert, seg_feature,kernal_verbs,adj_matrixes,)

                # 多卡
                # crf_loss = self.model.module.crf(emissions=pred_tags, mask=masks_crf, tags=seq_tags, reduction='mean')
                # 单卡
                crf_loss = self.model.crf(emissions=pred_tags, mask=masks_crf, tags=seq_tags, reduction='mean')
                tags_loss = cut_crossentropy_loss(pred_tags, seq_tags, self.device, cut_side=0.2,
                                                  weight_=self.cross_entropy_weight_)

                loss = tags_loss + crf_loss
                self.writer.set_step((epoch - 1) * len(self.valid_iter) + batch_idx, 'valid')
                self.valid_metrics.update('tags_loss', tags_loss.item())
                self.valid_metrics.update('crf_loss', crf_loss.item())
                self.valid_metrics.update('total_loss', loss.item())
                # 多卡
                # best_path = self.model.module.crf.decode(emissions=pred_tags, mask=masks_crf)
                # 单卡
                best_path = self.model.crf.decode(emissions=pred_tags, mask=masks_crf)
                X, Y, Z = self.evaluate(best_path, texts, arguments, text_map_seg_idxs, seg_idx_map_bert_idxs,
                                        bert_idx_map_seg_idxs, seg_idx_map_texts, raw_texts)
                X_all += X
                Y_all += Y
                Z_all += Z
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(X, Y, Z))
                recall = X / Z
                precision = X / Y
                f1 = 2 * X / (Y + Z)
                if epoch > -1 and f1 < 0.5:
                    # chance = math.tanh((0.5 - f1) / f1)
                    for text, path, gold_argument, text_map_seg_idx, seg_idx_map_bert_idx, bert_idx_map_seg_idx, seg_idx_map_text, raw_text in zip(
                            texts, best_path, arguments, text_map_seg_idxs, seg_idx_map_bert_idxs,
                            bert_idx_map_seg_idxs, seg_idx_map_texts, raw_texts):
                        pred_argument = bert_extract_arguments(text, path, self.schema, class_id=self.class_id,
                                                               text_map_seg_idx=text_map_seg_idx,
                                                               seg_idx_map_bert_idx=seg_idx_map_bert_idx,
                                                               bert_idx_map_seg_idx=bert_idx_map_seg_idx,
                                                               seg_idx_map_text=seg_idx_map_text,
                                                               raw_text=raw_text)
                        event_list = []
                        temp_argument = []
                        for k, v in pred_argument.items():
                            temp_argument.append({'role': v[1], 'argument': k})
                            event_list.append({
                                'event_type': v[0],
                                'arguments': [{
                                    'role': v[1],
                                    'argument': k
                                }]
                            })
                        # print('*'*20+'p:{},  r:{},  f:{}'.format(precision,recall,f1)+'*'*20)
                        # print('text:{}'.format(''.join(text)))
                        # print('event_list:{}'.format(event_list))
                        # print('gold_argument:{}'.format(str([{'role': v[1], 'argument': k.split('_')[0]} for k, v in gold_argument.items()])))
                        # print('pred_argument:{}'.format(str(temp_argument)))
                        # print('#'*40)
                        log_f.write('epoch:{}'.format(epoch)+'*'*20+'p:{},  r:{},  f:{}'.format(precision,recall,f1)+'*'*20+'\n')
                        log_f.write('text:{}'.format(''.join(text))+'\n')
                        # log_f.write('event_list:{}'.format(event_list)+'\n')
                        log_f.write('gold_argument:{}'.format(str([{'role': v[1], 'argument': k.split('_')[0]} for k, v in gold_argument.items()]))+'\n')
                        log_f.write('pred_argument:{}'.format(str(temp_argument))+'\n')
                        log_f.write('\n')
                        log_f.flush()
            P_all = X_all / Y_all
            R_all = X_all / Z_all
            F1_all = 2*X_all /(Y_all + Z_all)
        self.valid_metrics.update('P_all', P_all)
        self.valid_metrics.update('R_all', R_all)
        self.valid_metrics.update('F_all', F1_all)
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_iter, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def evaluate(self, batch_pred_tag, batch_text, batch_arguments, text_map_seg_idxs, seg_idx_map_bert_idxs,
                 bert_idx_map_seg_idxs, seg_idx_map_texts, raw_texts):
        """评测函数（跟官方评测结果不一定相同，但很接近）
        """

        X, Y, Z = 1e-10, 1e-10, 1e-10

        for pred_tag, text, arguments, text_map_seg_idx, seg_idx_map_bert_idx, bert_idx_map_seg_idx, seg_idx_map_text, raw_text in zip(
                batch_pred_tag, batch_text, batch_arguments, text_map_seg_idxs, seg_idx_map_bert_idxs,
                bert_idx_map_seg_idxs, seg_idx_map_texts, raw_texts):

            inv_arguments_label = {v: k for k, v in arguments.items()}
            pred_arguments = bert_extract_arguments(text, pred_tag, self.schema, class_id=self.class_id,
                                                    text_map_seg_idx=text_map_seg_idx,
                                                    seg_idx_map_bert_idx=seg_idx_map_bert_idx,
                                                    bert_idx_map_seg_idx=bert_idx_map_seg_idx,
                                                    seg_idx_map_text=seg_idx_map_text, raw_text=raw_text)
            pred_inv_arguments = {v: k for k, v in pred_arguments.items()}

            Y += len(pred_inv_arguments)

            Z += len(inv_arguments_label)
            for k, v in pred_inv_arguments.items():
                if k in inv_arguments_label:
                    argument_str = inv_arguments_label[k].split('_')[0]
                    # 用最长公共子串作为匹配程度度量
                    l = pylcs.lcs(v, argument_str)
                    # X += 2. * l / (len(v) + len(inv_arguments_label[k]))
                    y = len(v)
                    p = l / y+0.000001
                    z=len(argument_str)
                    r = l / z + 0.000001
                    f1 = 2*p*r/(p+r)
                    X += f1
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        return X, Y, Z

    def class_evaluate(self, class_pre, class_target):
        class_pre = class_pre.cpu().detach().numpy()
        class_pre = np.where(class_pre > 0.5, 1, 0)
        p = precision_score(class_target, class_pre, average='micro')
        r = recall_score(class_target, class_pre, average='micro')
        f1 = f1_score(class_target, class_pre, average='micro')
        return p, r, f1
