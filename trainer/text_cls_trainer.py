# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 11:20 下午
# @Author  : lizhen
# @FileName: text_cls_trainer.py
# @Description:
import numpy as np
import torch
from base.base_trainer import BaseTrainer
from utils.utils import inf_loop, MetricTracker, bert_extract_arguments
from time import time
import pylcs
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score


class TextCLSTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, train_iter, valid_iter, device, schema,
                 test_iter=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
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
        self.train_metrics = MetricTracker('p', 'r', 'f', 'loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('p', 'r', 'f', 'loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # self.cross_entropy_weight_ = [1.0] * schema.class_tag_num[class_id]
        # for i in range(1, schema.class_tag_num[class_id]):
        #     if i % 2 == 1:
        #         self.cross_entropy_weight_[i] = 1.5
        # self.cross_entropy_weight_[0] = 0.01

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        t1 = time()
        self.model.train()
        # self.model.crf.train()
        self.train_metrics.reset()
        for batch_idx, batch_data in enumerate(self.train_iter):
            self.optimizer.zero_grad()

            text_ids, seq_lens, masks_bert, texts,class_labels = batch_data
            pred_cls = self.model(text_ids, seq_lens, masks_bert)

            loss = F.binary_cross_entropy(pred_cls, class_labels)
            loss.backward()
            self.optimizer.step()
            p, r, f1 = self.class_evaluate(pred_cls, class_labels)
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('p', p)
            self.train_metrics.update('r', r)
            self.train_metrics.update('f', f1)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
            # 避免挤爆显存
            del seq_lens, masks_bert, texts, text_ids
            # torch.cuda.empty_cache()
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        print('spending time:', time() - t1)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_iter):
                text_ids, seq_lens, masks_bert, texts,class_labels = batch_data

                # context, seq_len, mask_bert,mask_crf,class_label,event_label,seq_tags
                pred_cls = self.model(text_ids, seq_lens, masks_bert)


                loss = F.binary_cross_entropy(pred_cls,class_labels)
                p, r, f1 = self.class_evaluate(pred_cls,class_labels)
                self.writer.set_step((epoch - 1) * len(self.valid_iter) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('p', p)
                self.valid_metrics.update('r', r)
                self.valid_metrics.update('f', f1)


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


    def class_evaluate(self, class_pre, class_target):
        class_pre = class_pre.detach().cpu().numpy()
        class_target = class_target.cpu().numpy()
        class_pre = np.where(class_pre > 0.5, 1, 0)
        p = precision_score(class_target, class_pre, average='micro')
        r = recall_score(class_target, class_pre, average='micro')
        f1 = f1_score(class_target, class_pre, average='micro')
        return p, r, f1
