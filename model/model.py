# -*- coding: utf-8 -*-
# @Time    : 2020/4/6 3:24 下午
# @Author  : lizhen
# @FileName: model.py
# @Description:
from torch import nn
import torch
from base.base_model import BaseModel
from transformers import BertModel


class RnnModel(BaseModel):
    def __init__(self, rnn_type, word_embedding, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, batch_first=False, use_pretrain_embedding=False):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        vocab_size, embedding_dim = len(word_embedding.stoi), word_embedding.vectors.shape[1]
        pad_index = word_embedding.stoi['PAD']
        if use_pretrain_embedding:
            self.embedding = nn.Embedding.from_pretrained(word_embedding.vectors, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2 if self.bidirectional else 1, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def prepare_pack_padded_sequence(self, inputs_words, seq_lengths, descending=True):
        """
        :param device:
        :param inputs_words:
        :param seq_lengths:
        :param descending:
        :return:
        """
        sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)

        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_inputs_words = inputs_words[indices]
        return sorted_inputs_words, sorted_seq_lengths, desorted_indices

    def forward(self, text, text_lengths):
        # text = [batch size,sent len ]
        text, sorted_seq_lengths, desorted_indices = self.prepare_pack_padded_sequence(text, text_lengths)
        embedded = self.dropout(self.embedding(text))

        # embedded = [ batch size,sent len, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=self.batch_first)

        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # hidden = [bidirection * n_layers,batch size , hidden dim]

        # unpack sequence
        # output = [ batch size,sent len, hidden_dim * bidirectional]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]
        # if not self.bidirectional:
        #     hidden = torch.reshape(hidden,(hidden.shape[1],self.hidden_dim * self.n_layers))
        # else:
        #     hidden = torch.reshape(hidden, (-1,hidden.shape[1], self.hidden_dim * self.n_layers))
        #     hidden = torch.mean(hidden,dim=0)

        out = self.fc(self.dropout(output))
        # out = self.fc(output)

        return out


class Bert(nn.Module):

    def __init__(self, bert_path, bert_train, num_classes):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # 对bert进行训练
        for name,param in self.bert.named_parameters():
            # if 'encoder.layer.11' in name:
            #     param.requires_grad = bert_train
            # else:
            #     param.requires_grad = False
            param.requires_grad = bert_train

        self.fc = nn.Linear(self.bert.config.to_dict()['hidden_size'], num_classes)

    def forward(self, context, seq_len, mask):
        # context  输入的句子序列
        # seq_len  句子长度
        # mask     对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        # cls [batch_size, 768]
        # sentence [batch size,sen len,  768]
        batch_size = context.shape[0]
        context = torch.reshape(context, [-1, context.shape[-1]])
        mask = torch.reshape(mask, [-1, mask.shape[-1]])

        sentence, cls = self.bert(context, attention_mask=mask)
        out = self.fc(sentence)
        out_ = out[:, 1:, :]
        return out_


class BertRNN(nn.Module):

    def __init__(self, rnn_type, bert_path, bert_train, hidden_dim, n_layers, bidirectional, batch_first, dropout,
                 num_classes, bert_embedding_dim=768):
        super(BertRNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bert = BertModel.from_pretrained(bert_path)
        # 对bert进行训练
        for name,param in self.bert.named_parameters():
            # if 'encoder.layer.11' in name:
            #     param.requires_grad = bert_train
            # else:
            #     param.requires_grad = False
            param.requires_grad=bert_train
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(bert_embedding_dim,
                               hidden_size=hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(bert_embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(bert_embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * n_layers, num_classes)

    def forward(self, context, seq_len, mask):
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        batch_size = context.shape[0]
        context = torch.reshape(context, [-1, context.shape[-1]])
        mask = torch.reshape(mask, [-1, mask.shape[-1]])

        encoder_out, text_cls = self.bert(context, attention_mask=mask)

        encoder_out = torch.reshape(encoder_out, [batch_size, -1, 768])

        encoder_out, sorted_seq_lengths, desorted_indices = self.prepare_pack_padded_sequence(encoder_out, seq_len)

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(encoder_out, sorted_seq_lengths,
                                                            batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # unpack sequence
        # output = [ batch size,sent len, hidden_dim * bidirectional]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]
        #
        # if not self.bidirectional:
        #     hidden = torch.reshape(hidden,(hidden.shape[1],self.hidden_dim * self.n_layers))
        # else:
        #     hidden = torch.reshape(hidden, (-1,hidden.shape[1], self.hidden_dim * self.n_layers))
        #     hidden = torch.mean(hidden,dim=0)
        # output = torch.sum(output,dim=1)
        out = self.fc(self.dropout(output))

        return out[:,1:,:]

    def prepare_pack_padded_sequence(self, inputs_words, seq_lengths, descending=True):
        """
        :param device:
        :param inputs_words:
        :param seq_lengths:
        :param descending:
        :return:
        """
        sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)

        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_inputs_words = inputs_words[indices]
        return sorted_inputs_words, sorted_seq_lengths, desorted_indices
