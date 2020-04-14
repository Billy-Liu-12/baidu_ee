# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 5:47 下午
# @Author  : lizhen
# @FileName: trainer.py
# @Description:
import numpy as np
import torch
from base.base_trainer import BaseTrainer
from utils.utils import inf_loop, MetricTracker
from time import time
import pylcs

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model,crf_model, criterion, metric_ftns, optimizer, config, train_iter, valid_iter,schema, test_iter=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
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
            pred_arguments = self.extract_arguments(text,pred_tag)
            pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
            Y += len(pred_inv_arguments)
            Z += len(inv_arguments)
            for k, v in pred_inv_arguments.items():
                if k in inv_arguments:
                    # 用最长公共子串作为匹配程度度量
                    l = pylcs.lcs(v, inv_arguments[k])
                    X += 2. * l / (len(v) + len(inv_arguments[k]))
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return  X,Y,Z

    def extract_arguments(self,text,pred_tag):
        """arguments抽取函数
        """
        arguments, starting = [], False
        for i, label in enumerate(pred_tag):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    index = self.schema.id2role[(label - 1) // 2].rfind('-')
                    arguments.append([[i], (self.schema.id2role[(label - 1) // 2][:index],self.schema.id2role[(label - 1) // 2][index+1:])])
                elif starting:
                    arguments[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return {text[idx[0]:idx[-1]]: l for idx, l in arguments }


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

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'crf_state_dict':self.crf_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")


class BertTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, train_iter, valid_iter, test_iter=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
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
        self.train_metrics.reset()
        for batch_idx, batch_data in enumerate(self.train_iter):
            (x, seq_len, mask), y = batch_data
            self.optimizer.zero_grad()
            output = self.model(x, seq_len, mask).squeeze(1)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, y))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
            # 避免挤爆显存
            del x, seq_len,mask,y, loss
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
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_iter):
                (x, seq_len, mask), y = batch_data

                output = self.model(x, seq_len, mask).squeeze(1)
                loss = self.criterion(output, y)

                self.writer.set_step((epoch - 1) * len(self.valid_iter) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, y))

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
