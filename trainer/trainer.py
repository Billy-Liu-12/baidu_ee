# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 5:47 下午
# @Author  : lizhen
# @FileName: trainer.py
# @Description:
import numpy as np
import torch
from base.base_trainer import BaseTrainer
from utils.utils import inf_loop, MetricTracker,extract_arguments,bert_extract_arguments
from time import time
import pylcs
import torch.nn.functional as F

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model,crf_model, criterion, metric_ftns, optimizer, config, train_iter, valid_iter,schema, test_iter=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model,crf_model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.schema = schema
        self.crf_model = crf_model
        self.train_iter, self.valid_iter, self.test_iter = train_iter, valid_iter, test_iter
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_iter)
        else:
            # iteration-based training
            self.data_loader = inf_loop(train_iter)
            self.len_epoch = len_epoch

        self.do_validation = self.valid_iter is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_iter.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        t1 = time()
        self.model.train()
        self.crf_model.train()
        self.train_metrics.reset()
        for batch_idx, batch_data in enumerate(self.train_iter):
            self.optimizer.zero_grad()

            text_ids, seq_lens, masks,raw_text,raw_arguments,seq_tags = batch_data

            output = self.model(text_ids, seq_lens)
            loss = self.criterion(output, seq_tags)
            loss += -self.crf_model(emissions=output, mask=masks, tags=seq_tags)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            best_path = self.crf_model.decode(emissions=output, mask=masks)
            X,Y,Z = self.evaluate(best_path,raw_text,raw_arguments)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(X,Y,Z))


            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        print('spending time:', time() - t1)
        return log

    def evaluate(self,batch_pred_tag,batch_text,batch_arguments):
        """评测函数（跟官方评测结果不一定相同，但很接近）
        """

        X, Y, Z = 1e-10, 1e-10, 1e-10

        for pred_tag,text, arguments in zip(batch_pred_tag,batch_text,batch_arguments):

            inv_arguments = {v: k for k, v in arguments.items()}
            pred_arguments = extract_arguments(text,pred_tag,self.schema)
            pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
            Y += len(pred_inv_arguments)
            Z += len(inv_arguments)
            for k, v in pred_inv_arguments.items():
                if k in inv_arguments:
                    # 用最长公共子串作为匹配程度度量
                    l = pylcs.lcs(v, inv_arguments[k])
                    X += 2. * l / (len(v) + len(inv_arguments[k]))
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return X,Y,Z




    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.crf_model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_iter):
                text_ids, seq_lens, masks,raw_text,raw_arguments,seq_tags = batch_data
                output = self.model(text_ids, seq_lens)
                loss = self.criterion(output, seq_tags)
                loss += -self.crf_model(emissions=output, mask=masks, tags=seq_tags)

                self.writer.set_step((epoch - 1) * len(self.valid_iter) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                best_path = self.crf_model.decode(emissions=output, mask=masks)
                X, Y, Z = self.evaluate(best_path, raw_text, raw_arguments)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(X, Y, Z))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        for name, p in self.crf_model.named_parameters():
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








class BertTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, crf_model,criterion, metric_ftns, optimizer, config, train_iter, valid_iter, schema,test_iter=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model,crf_model, criterion, metric_ftns, optimizer, config)
        self.config = config
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

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        t1 = time()
        self.model.train()
        self.crf_model.train()
        self.train_metrics.reset()
        for batch_idx, batch_data in enumerate(self.train_iter):
            sentence_feature,text_ids, seq_lens, masks_bert, masks_crf, texts, arguments,class_labels,event_labels, seq_tags= batch_data
            class_labels = torch.LongTensor(np.array(class_labels)).cuda()
            event_labels = torch.LongTensor(np.array(event_labels)).cuda()
            self.optimizer.zero_grad()
            out_class,out_event,output = self.model(sentence_feature, seq_lens)

            loss = F.cross_entropy(out_class,class_labels)
            loss += F.cross_entropy(out_event,event_labels)
            loss += self.criterion(output, seq_tags)

            loss += -self.crf_model(emissions=output, mask=masks_crf, tags=seq_tags)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            best_path = self.crf_model.decode(emissions=output, mask=masks_crf)

            X, Y, Z = self.evaluate(best_path, texts, arguments)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(X, Y, Z))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
            # 避免挤爆显存
            # del x, seq_len,mask,y, loss
            torch.cuda.empty_cache()
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
        self.crf_model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_iter):
                sentence_feature,text_ids, seq_lens, masks_bert, masks_crf, texts, arguments,class_labels,event_labels, seq_tags= batch_data

                out_class, out_event, output = self.model(sentence_feature, seq_lens)
                loss = F.cross_entropy(out_class, torch.from_numpy(np.array(class_labels)).cuda())
                loss += F.cross_entropy(out_event, torch.from_numpy(np.array(event_labels)).cuda())
                loss += self.criterion(output, seq_tags)
                loss += -self.crf_model(emissions=output, mask=masks_crf, tags=seq_tags)

                self.writer.set_step((epoch - 1) * len(self.valid_iter) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                best_path = self.crf_model.decode(emissions=output, mask=masks_crf)
                X, Y, Z = self.evaluate(best_path, texts, arguments)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(X, Y, Z))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        for name, p in self.crf_model.named_parameters():
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

    def evaluate(self,batch_pred_tag,batch_text,batch_arguments):
        """评测函数（跟官方评测结果不一定相同，但很接近）
        """

        X, Y, Z = 1e-10, 1e-10, 1e-10

        for pred_tag,text, arguments in zip(batch_pred_tag,batch_text,batch_arguments):

            inv_arguments_label = {v: k for k, v in arguments.items()}
            pred_arguments = bert_extract_arguments(text,pred_tag,self.schema)
            pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
            Y += len(pred_inv_arguments)
            Z += len(inv_arguments_label)
            for k, v in pred_inv_arguments.items():
                if k in inv_arguments_label:
                    # 用最长公共子串作为匹配程度度量
                    l = pylcs.lcs(v, inv_arguments_label[k])
                    X += 2. * l / (len(v) + len(inv_arguments_label[k]))
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        return X,Y,Z

    def extract_arguments(self,text, pred_tag, schema):
        """arguments抽取函数
        """
        arguments, starting = [], False
        for i, label in enumerate(pred_tag):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    index = schema.id2role[(label - 1) // 2].rfind('-')
                    arguments.append([[i], (
                        schema.id2role[(label - 1) // 2][:index], schema.id2role[(label - 1) // 2][index + 1:])])
                elif starting:
                    arguments[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return {''.join(text[idx[0]:idx[-1] + 1]).replace('#', '').replace('[UNK]', '').strip(): l for idx, l in
                arguments}

