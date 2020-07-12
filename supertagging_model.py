from __future__ import absolute_import, division, print_function

import os

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from pytorch_pretrained_bert.modeling import (BertPreTrainedModel, BertModel)
from pytorch_pretrained_bert.tokenization import BertTokenizer

from torch.nn import CrossEntropyLoss, Softmax
from supertagging_helper import read_tsv


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class GCNModule(nn.Module):
    def __init__(self, layer_number, hidden_size, use_weight=False, output_all_layers=False):
        super(GCNModule, self).__init__()
        if layer_number < 1:
            raise ValueError()
        self.layer_number = layer_number
        self.output_all_layers = output_all_layers
        self.GCNLayers = nn.ModuleList(([GCNLayer(hidden_size, use_weight)
                                         for _ in range(self.layer_number)]))

    def forward(self, hidden_state, adjacency_matrix):
        # hidden_state = self.first_GCNLayer(hidden_state, adjacency_matrix, type_seq, type_matrix)
        # all_output_layers.append(hidden_state)

        all_output_layers = []

        for gcn in self.GCNLayers:
            hidden_state = gcn(hidden_state, adjacency_matrix)
            all_output_layers.append(hidden_state)

        if self.output_all_layers:
            return all_output_layers
        else:
            return all_output_layers[-1]


class GCNLayer(nn.Module):
    def __init__(self, hidden_size, use_weight=False):
        super(GCNLayer, self).__init__()
        self.temper = hidden_size ** 0.5
        self.use_weight = use_weight
        self.relu = nn.ReLU()

        self.linear = nn.Linear(hidden_size, hidden_size)

        if self.use_weight:
            self.left_linear = nn.Linear(hidden_size, hidden_size, bias=False)
            self.right_linear = nn.Linear(hidden_size, hidden_size, bias=False)
            self.self_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.left_linear = None
            self.right_linear = None
            self.self_linear = None

        self.output_layer_norm = LayerNormalization(hidden_size)

    def get_att(self, matrix_1, matrix_2, adjacency_matrix):

        if self.use_weight:
            m_left = self.left_linear(matrix_2)
            m_self = self.left_linear(matrix_2)
            m_right = self.right_linear(matrix_2)

            m_left = m_left.permute(0, 2, 1)
            m_self = m_self.permute(0, 2, 1)
            m_right = m_right.permute(0, 2, 1)

            u_left = torch.matmul(matrix_1, m_left) / self.temper
            u_self = torch.matmul(matrix_1, m_self) / self.temper
            u_right = torch.matmul(matrix_1, m_right) / self.temper

            adj_tri = torch.triu(adjacency_matrix, diagonal=1)
            u_left = torch.mul(u_left, adj_tri)
            u_self = torch.mul(u_self, torch.triu(adjacency_matrix, diagonal=0) - adj_tri)
            u_right = torch.mul(u_right, adj_tri.permute(0, 2, 1))

            u = u_left + u_right + u_self

            exp_u = torch.exp(u)
            delta_exp_u = torch.mul(exp_u, adjacency_matrix)
        else:
            delta_exp_u = adjacency_matrix

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        attention = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)
        return attention

    def forward(self, hidden_state, adjacency_matrix):
        # hidden_state: (batch_size, character_seq_len, hidden_size)
        # adjacency_matrix: (batch_size, character_seq_len_1, character_seq_len_2)
        # type_seq: (batch_size, type_seq_len)
        # type_matrix: (batch_size, character_seq_len_1, type_seq_len)

        # tmp_hidden = hidden_state.permute(0, 2, 1)
        context_attention = self.get_att(hidden_state, hidden_state, adjacency_matrix)

        hidden_state = self.linear(hidden_state)
        context_attention = torch.bmm(context_attention, hidden_state)

        o = self.output_layer_norm(context_attention)

        # o = context_attention
        output = self.relu(o)

        return output


class Supertagger(nn.Module):

    def __init__(self, labelmap, args, gram2id=None, tag2id=None, type2id=None, strong_segments=None):
        super(Supertagger, self).__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec['args'] = args
        self.tag2id = tag2id
        self.gram2id = gram2id
        self.type2id = type2id
        self.strong_segments = strong_segments

        self.labelmap = labelmap
        self.id2label = {v: k for k, v in self.labelmap.items()}
        self.num_labels = len(self.labelmap)
        self.max_seq_length = args.max_seq_length
        self.max_ngram_size = args.max_ngram_size
        self.max_ngram_length = args.max_ngram_length

        self.gcn_layer_number = args.gcn_layer_number
        self.use_weight = args.use_weight

        self.window_size = args.window_size
        self.clipping_top_n = args.clipping_top_n
        self.clipping_threshold = args.clipping_threshold

        self.bert_tokenizer = None
        self.bert = None
        self.xlnet_tokenizer = None
        self.xlnet = None

        if args.use_bert:
            cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                           'distributed_{}'.format(args.local_rank))
            self.bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
            self.bert = BertModel.from_pretrained(args.bert_model, cache_dir=cache_dir)
            self.hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif args.use_xlnet:
            raise ValueError()
            # from pytorch_transformers import XLNetModel, XLNetTokenizer
            # self.xlnet_tokenizer = XLNetTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
            # self.xlnet = XLNetModel.from_pretrained(args.bert_model)
            # hidden_size = self.bert.config.hidden_size
            # self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        else:
            raise ValueError()

        self.classifier = nn.Linear(self.hidden_size, self.num_labels, bias=False)

        self.gcn = GCNModule(self.gcn_layer_number, self.hidden_size,
                             use_weight=self.use_weight,
                             output_all_layers=False)

        self.loss_fct = CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, word_seq=None, word_mask=None, tag_seq=None, tag_matrix=None,
                adjacency_matrix=None, type_seq=None, type_matrix=None):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.xlnet is not None:
            # sequence_output, _ = self.xlnet()
            raise ValueError()
        else:
            raise ValueError()

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp

        sequence_output = self.dropout(valid_output)

        sequence_output = self.gcn(sequence_output, adjacency_matrix)

        # sequence_output = self.test_linear(sequence_output)
        # sequence_output = nn.ReLU()(sequence_output)
        logits = self.classifier(sequence_output)
        # logits = self.softmax(logits)
        total_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        assert not torch.isnan(total_loss)

        return total_loss, logits

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model):
        spec = spec.copy()
        res = cls(**spec)
        res.load_state_dict(model)
        return res

    def load_data(self, data_path, flag):
        lines = read_tsv(data_path)

        overlap_num = 0

        data = []

        for sentence, label, pos_tags, governor_index, relation_type in lines:
            word_list = []
            word_matching_position = []
            tag_list = []
            tag_matching_position = []

            if len(sentence) > self.max_seq_length:
                continue

            word_list = None
            word_matching_position = None
            tag_list = None
            tag_matching_position = None

            type_matching = []

            type_list = None

            s = [sentence[0]]
            n_b = []
            # # use strong segments
            for i in range(len(sentence) - 1):
                ngram = (sentence[i].lower(), sentence[i+1].lower())
                if ngram in self.strong_segments:
                    s.append(sentence[i + 1])
                else:
                    n_b.append((i+1-len(s), i))
                    s = [sentence[i + 1]]
            if len(s) > 0:
                n_b.append((len(sentence) - len(s), len(sentence) - 1))

            ngram_index = []
            for s, e in n_b:
                if not s == e:
                    for i in range(s, e+1):
                        for j in range(s, e+1):
                            if not i == j:
                                type_matching.append((i, j, None))
                    ngram_index.append((s, e))

            for i in range(len(ngram_index) - 1):
                s_1 = ngram_index[i]
                s_2 = ngram_index[i+1]
                type_matching.append((s_1[0], s_2[0], None))
                type_matching.append((s_1[0], s_2[1], None))
                type_matching.append((s_1[1], s_2[0], None))
                type_matching.append((s_1[1], s_2[1], None))

            data.append((sentence, label, word_list, word_matching_position, tag_list, tag_matching_position,
                         type_list, type_matching))

        examples = []
        for i, (sentence, label, word_list, word_matching_position, tag_list, tag_matching_position,
                type_list, type_matching) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            word = word_list
            label = label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             word=word, word_matrix=word_matching_position,
                             tag=tag_list, tag_matrix=tag_matching_position,
                             type_list=type_list, type_matrix=type_matching))
        return examples

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputBatch`s."""

        features = []

        # -------- max ngram size --------
        max_seq_length = 0
        max_word_size = 0
        max_tag_size = 0
        # -------- max ngram size --------

        if self.bert is not None:
            tokenizer = self.bert_tokenizer
        elif self.xlnet is not None:
            tokenizer = self.xlnet_tokenizer
        else:
            raise ValueError()

        all_tokens = []
        all_labels = []
        all_valid = []
        all_label_mask = []
        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            tokens = []
            labels = []
            valid = []
            label_mask = []

            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)

            if len(tokens) > max_seq_length:
                max_seq_length = len(tokens)
            all_tokens.append(tokens)
            all_labels.append(labels)
            all_valid.append(valid)
            all_label_mask.append(label_mask)

        max_seq_length += 2

        if max_seq_length > 510:
            print('')

        for (example, tokens, labels, valid, label_mask) \
                in zip(examples, all_tokens, all_labels, all_valid, all_label_mask):
            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            label_ids.append(self.labelmap["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    if labels[i] in self.labelmap:
                        label_ids.append(self.labelmap[labels[i]])
                    else:
                        label_ids.append(self.labelmap['<UNK>'])
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(self.labelmap["[SEP]"])

            # segment_id: [0, 0, ..., 0] length 7
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            # input_ids: [1, 2, 3, .. , 7] length 7
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            type_matching_position = example.type_matrix

            adjacency_matrix = np.zeros((max_seq_length, max_seq_length), dtype=np.int)
            for (i, j, _) in type_matching_position:
                adjacency_matrix[i+1][j+1] = 1
                adjacency_matrix[j+1][i+1] = 1
            # add self
            for i in range(len(adjacency_matrix)):
                adjacency_matrix[i][i] = 1
            # add link between words
            for i in range(len(adjacency_matrix) - 1):
                adjacency_matrix[i][i+1] = 1
                adjacency_matrix[i+1][i] = 1

            type_index = None
            word_ids = None
            word_matching_matrix = None
            tag_ids = None
            tag_matching_matrix = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              word_ids=word_ids,
                              word_matching_matrix=word_matching_matrix,
                              tag_ids=tag_ids,
                              tag_matching_matrix=tag_matching_matrix,
                              dep_type_ids=type_index,
                              dep_adjacency_matrix=adjacency_matrix))
        return features


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 word=None, word_matrix=None, tag=None, tag_matrix=None,
                 type_list=None, type_matrix=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.word = word
        self.word_matrix = word_matrix
        self.tag = tag
        self.tag_matrix = tag_matrix
        self.type_list = type_list
        self.type_matrix = type_matrix


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, word_ids=None,
                 word_matching_matrix=None, tag_ids=None, tag_matching_matrix=None,
                 dep_type_ids=None, dep_adjacency_matrix=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.word_ids = word_ids
        self.word_matching_matrix = word_matching_matrix
        self.tag_ids = tag_ids
        self.tag_matching_matrix = tag_matching_matrix
        self.dep_type_ids = dep_type_ids
        self.dep_adjacency_matrix = dep_adjacency_matrix
