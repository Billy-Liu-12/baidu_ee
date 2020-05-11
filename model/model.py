# -*- coding: utf-8 -*-
# @Time    : 2020/4/6 3:24 下午
# @Author  : lizhen
# @FileName: model.py
# @Description:
from torch import nn
import torch
from base.base_model import BaseModel
from transformers import BertModel
# from .torch_crf import CRF
from .torch_crf_r import CRF
import numpy as np
import torch.nn.functional as F


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


class Bert_Text_Cls(nn.Module):

    def __init__(self, bert_path, bert_train, num_classes, dropout):
        super(Bert_Text_Cls, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # 对bert进行训练
        for name, param in self.bert.named_parameters():
            param.requires_grad = bert_train
        self.att_class_classifier = multi_class_Attention(self.bert.config.to_dict()['hidden_size'], num_classes)
        self.fc = nn.Linear(self.bert.config.to_dict()['hidden_size'], num_classes)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, context, seq_len, mask):
        # context  输入的句子序列
        # seq_len  句子长度
        # mask     对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        # cls [batch_size, 768]
        # sentence [batch size,sen len,  768]
        batch_size = context.shape[0]

        bert_sentence, bert_cls = self.bert(context, attention_mask=mask)
        output, weights = self.att_class_classifier(bert_sentence)
        pred_class = self.sigmoid(output)
        return pred_class


class Bert_CRF(BaseModel):

    def __init__(self, bert_path, bert_train, num_tags, seg_vocab_size, dropout, restrain):
        super(Bert_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.crf = CRF(num_tags, batch_first=True, restrain_matrix=restrain, loss_side=2.5)
        # 对bert进行训练
        for name, param in self.bert.named_parameters():
            param.requires_grad = bert_train

        # self.seg_embedding = nn.Embedding(seg_vocab_size, 10)
        self.fc_tags = nn.Linear(self.bert.config.to_dict()['hidden_size'], num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, seq_len, mask_bert, seg_feature):
        # context  输入的句子序列
        # seq_len  句子长度
        # mask     对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        context = torch.reshape(context, [-1, context.shape[-1]])
        mask_bert = torch.reshape(mask_bert, [-1, mask_bert.shape[-1]])

        # embedding
        # seg_embedding = self.dropout(self.seg_embedding(seg_feature))

        # cls [batch_size, 768]
        # sentence [batch size,sen len,  768]
        bert_sentence, bert_cls = self.bert(context, attention_mask=mask_bert)
        sentence_len = bert_sentence.shape[1]

        bert_cls = bert_cls.unsqueeze(dim=1).repeat(1, sentence_len, 1)
        bert_sentence = bert_sentence + bert_cls
        pred_tags = self.fc_tags(self.dropout(bert_sentence))[:, 1:, :]
        return pred_tags


class multi_class_Attention(nn.Module):
    def __init__(self, hidden_dim, class_num):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 256),  # query
            nn.ReLU(True),
            nn.Linear(256, class_num)  # class key
        )

        class_weight = torch.randn((class_num, hidden_dim), requires_grad=True)
        self.class_weight = torch.nn.Parameter(class_weight)
        class_bias = torch.randn(class_num, requires_grad=True)
        self.class_bias = torch.nn.Parameter(class_bias)

    def forward(self, encoder_outputs, class_feature=None):
        # (B, L, H) -> (B , L, class num)
        energy = self.projection(encoder_outputs)
        # (B , L, class num)
        weights = F.softmax(energy, dim=1)
        # (B , class num, L)
        weights = weights.permute(0, 2, 1)
        # (B,, class num, L) * (B, L, H) -> (B,class_num , H)
        features = torch.bmm(weights, encoder_outputs)
        features = torch.nn.Dropout(0.5)(features)
        # (B, class_num,H)
        output = torch.sum(torch.mul(features, self.class_weight), -1) + self.class_bias

        return output, weights


class BertRNNCRF(nn.Module):
    def __init__(self, num_tags, rnn_type, bert_path, bert_train, hidden_dim, n_layers, bidirectional, batch_first,
                 dropout, restrain):
        super(BertRNNCRF, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bert = BertModel.from_pretrained(bert_path)
        # 对bert进行训练
        for name, param in self.bert.named_parameters():
            param.requires_grad = bert_train
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.bert.config.to_dict()['hidden_size'],
                               hidden_size=hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.crf = CRF(num_tags, batch_first=True, restrain_matrix=restrain, loss_side=3)
        self.dropout = nn.Dropout(dropout)
        if self.bidirectional:
            # self.fc_tags = nn.Linear(hidden_dim*2, num_tags)
            self.fc_tags = nn.Linear(256, num_tags)
        else:
            self.fc_tags = nn.Linear(hidden_dim, num_tags)

        self.fc_bert = nn.Linear(self.bert.config.to_dict()['hidden_size'], 256)

    def forward(self, context, seq_len, max_seq_len, mask_bert):
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        #
        # batch_size = context.shape[0]
        # context = torch.reshape(context, [-1, context.shape[-1]])
        # mask = torch.reshape(mask_bert, [-1, mask_bert.shape[-1]])
        bert_sentence, bert_cls = self.bert(context, attention_mask=mask_bert)
        sentence_len = bert_sentence.shape[1]

        bert_cls = bert_cls.unsqueeze(dim=1).repeat(1, sentence_len, 1)
        bert_sentence = bert_sentence + bert_cls
        bert_out = F.relu(self.fc_bert(bert_sentence))

        encoder_out, sorted_seq_lengths, desorted_indices = self.prepare_pack_padded_sequence(bert_sentence, seq_len)

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

        # output = torch.cat([bert_out,output],dim=2)
        out = self.fc_tags(self.dropout(output))

        return out[:, 1:, :]

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


class BertGCNCRF(nn.Module):
    def __init__(self, num_tags,event_label_num, bert_path, bert_train, dropout, restrain):
        super(BertGCNCRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # 对bert进行训练
        for name, param in self.bert.named_parameters():
            param.requires_grad = bert_train
        self.crf = CRF(num_tags, batch_first=True, restrain_matrix=restrain, loss_side=3)
        self.dropout = nn.Dropout(dropout)

        self.fc_gcn_in = nn.Linear(self.bert.config.to_dict()['hidden_size'], 256)
        self.fc_gcn_out = nn.Linear(256, 128)

        self.gcn_other_1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.gcn_other_2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.gcn_other_3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())

        self.gcn_subject_1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.gcn_subject_2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.gcn_subject_3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())

        self.gcn_object_1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.gcn_object_2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.gcn_object_3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())

        self.fc_bert = nn.Linear(self.bert.config.to_dict()['hidden_size'], 128)

        self.fc_tags = nn.Linear(256+128, num_tags)
        self.fc_event_label = nn.Linear(self.bert.config.to_dict()['hidden_size'],event_label_num)

    def forward(self, context, seq_len, max_seq_len, mask_bert, seg_feature, kernal_verbs, adj_matrixes_subjects,
                adj_matrixes_objects, adj_matrixes_others):
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        bert_sentence, bert_cls = self.bert(context, attention_mask=mask_bert, token_type_ids=kernal_verbs)

        pred_event = self.fc_event_label(bert_cls)
        sentence_len = bert_sentence.shape[1]
        bert_cls = bert_cls.unsqueeze(dim=1).repeat(1, sentence_len, 1)
        bert_sentence = bert_sentence + bert_cls

        ###################### GCN ####################################
        gcn_in = self.dropout(self.fc_gcn_in(bert_sentence))

        # 1
        hidden_other = self.gcn_other_1(torch.bmm(adj_matrixes_others, gcn_in))
        hidden_subject = self.gcn_subject_1(torch.bmm(adj_matrixes_subjects, gcn_in))
        hidden_object = self.gcn_object_1(torch.bmm(adj_matrixes_objects, gcn_in))
        gcn_output = self.dropout(hidden_object + hidden_other + hidden_subject + gcn_in)
        # 2
        hidden_other = self.gcn_other_2(torch.bmm(adj_matrixes_others, gcn_output))
        hidden_subject = self.gcn_subject_2(torch.bmm(adj_matrixes_subjects, gcn_output))
        hidden_object = self.gcn_object_2(torch.bmm(adj_matrixes_objects, gcn_output))
        gcn_output = self.dropout(hidden_object + hidden_other + hidden_subject + gcn_output)
        # 3
        hidden_other = self.gcn_other_3(torch.bmm(adj_matrixes_others, gcn_output))
        hidden_subject = self.gcn_subject_3(torch.bmm(adj_matrixes_subjects, gcn_output))
        hidden_object = self.gcn_object_3(torch.bmm(adj_matrixes_objects, gcn_output))

        gcn_hidden = hidden_object + hidden_other + hidden_subject

        bert_out = self.fc_bert(bert_sentence)
        out = self.dropout(self.fc_tags(torch.cat([gcn_hidden, self.fc_gcn_out(gcn_output),bert_out], dim=2)))

        return out[:, 1:, :],pred_event



class RNN(nn.Module):

    def __init__(self, rnn_type, hidden_dim, n_layers, bidirectional, batch_first, dropout,
                 num_classes, bert_embedding_dim=768):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first

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
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, sentence_feature, seq_len):
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        batch_size = sentence_feature.shape[0]
        encoder_out = torch.reshape(sentence_feature, [batch_size, -1, 768])

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

        return out[:, 1:, :]

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


class RNN_MultiOut(nn.Module):

    def __init__(self, rnn_type, hidden_dim, n_layers, bidirectional, batch_first, dropout,
                 num_classes, bert_embedding_dim=768):
        super(RNN_MultiOut, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first

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
        self.fc_class = nn.Linear(hidden_dim * 2, 10)
        self.fc_event = nn.Linear(hidden_dim * 2, 66)
        self.fc_tag = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, sentence_feature, seq_len):
        # context    输入的句子
        # mask  对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        batch_size = sentence_feature.shape[0]
        encoder_out = torch.reshape(sentence_feature, [batch_size, -1, 768])

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

        output_cls = torch.sum(output, dim=1)
        out_class = self.fc_class(self.dropout(output_cls))
        out_event = self.fc_event(self.dropout(output_cls))
        out_tag = self.fc_tag(self.dropout(output))

        return out_class, out_event, out_tag[:, 1:, :]

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
