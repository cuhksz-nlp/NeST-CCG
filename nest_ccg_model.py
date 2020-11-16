from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
from torch import nn

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from nest_ccg_helper import read_tsv, load_json, save_json
from torch.nn import CrossEntropyLoss
import subprocess

DEFAULT_HPARA = {
    'max_seq_length': 300,
    'use_bert': False,
    'do_lower_case': False,
    'use_weight': False,
    'use_gcn': False,
    'gcn_layer_number': 3,
    'max_ngram_length': 5,
    'use_in_chunk': False,
    'use_cross_chunk': False,

    # used for parsing
    'clipping_top_n': 5,
    'clipping_threshold': 0.0005,
}

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


class NeSTCCG(nn.Module):

    def __init__(self, labelmap, hpara, model_path, gram2id=None, type2id=None):
        super(NeSTCCG, self).__init__()

        self.gram2id = gram2id
        self.type2id = type2id

        self.hpara = hpara

        self.labelmap = labelmap
        self.id2label = {v: k for k, v in self.labelmap.items()}
        self.num_labels = len(self.labelmap)
        self.max_seq_length = self.hpara['max_seq_length']

        self.gcn_layer_number = self.hpara['gcn_layer_number']
        self.use_weight = self.hpara['use_weight']
        self.max_ngram_length = self.hpara['max_ngram_length']
        self.use_in_chunk = self.hpara['use_in_chunk']
        self.use_cross_chunk = self.hpara['use_cross_chunk']

        self.clipping_top_n = self.hpara['clipping_top_n']
        self.clipping_threshold = self.hpara['clipping_threshold']

        self.bert_tokenizer = None
        self.bert = None
        self.xlnet_tokenizer = None
        self.xlnet = None

        if self.hpara['use_bert']:
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.bert = BertModel.from_pretrained(model_path, cache_dir='')
            self.hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        else:
            raise ValueError()

        if self.hpara['use_gcn']:
            self.gcn = GCNModule(self.gcn_layer_number, self.hidden_size,
                                 use_weight=self.use_weight,
                                 output_all_layers=False)
        else:
            self.gcn = None

        self.classifier = nn.Linear(self.hidden_size, self.num_labels, bias=False)

        self.loss_fct = CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None,
                adjacency_matrix=None):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        else:
            raise ValueError()

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp

        sequence_output = self.dropout(valid_output)

        if self.gcn is not None:
            sequence_output = self.gcn(sequence_output, adjacency_matrix)

        logits = self.classifier(sequence_output)

        if labels is None:
            return logits
        else:
            total_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            assert not torch.isnan(total_loss)
            return total_loss

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['use_weight'] = args.use_weight
        hyper_parameters['use_gcn'] = args.use_gcn
        hyper_parameters['gcn_layer_number'] = args.gcn_layer_number
        hyper_parameters['max_ngram_length'] = args.max_ngram_length
        hyper_parameters['use_in_chunk'] = args.use_in_chunk
        hyper_parameters['use_cross_chunk'] = args.use_cross_chunk

        hyper_parameters['clipping_top_n'] = args.clipping_top_n
        hyper_parameters['clipping_threshold'] = args.clipping_threshold

        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def load_model(cls, model_path):
        label_map = load_json(os.path.join(model_path, 'label_map.json'))
        hpara = load_json(os.path.join(model_path, 'hpara.json'))

        gram2id_path = os.path.join(model_path, 'gram2id.json')
        gram2id = load_json(gram2id_path) if os.path.exists(gram2id_path) else None
        gram2id = {tuple(k.split('`')): v for k, v in gram2id.items()}

        type2id_path = os.path.join(model_path, 'type2id.json')
        type2id = load_json(type2id_path) if os.path.exists(type2id_path) else None

        res = cls(model_path=model_path, labelmap=label_map, hpara=hpara,
                  gram2id=gram2id, type2id=type2id)
        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))
        return res

    def save_model(self, output_dir, vocab_dir):
        output_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        label_map_file = os.path.join(output_dir, 'label_map.json')

        if not os.path.exists(label_map_file):
            save_json(label_map_file, self.labelmap)

            save_json(os.path.join(output_dir, 'hpara.json'), self.hpara)
            if self.gram2id is not None:
                gram2save = {'`'.join(list(k)): v for k, v in self.gram2id.items()}
                save_json(os.path.join(output_dir, 'gram2id.json'), gram2save)
            if self.type2id is not None:
                save_json(os.path.join(output_dir, 'type2id.json'), self.type2id)

            output_config_file = os.path.join(output_dir, 'config.json')
            with open(output_config_file, "w", encoding='utf-8') as writer:
                if self.bert:
                    writer.write(self.bert.config.to_json_string())
                else:
                    raise ValueError()
            output_bert_config_file = os.path.join(output_dir, 'bert_config.json')
            command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
            subprocess.run(command, shell=True)

            if self.bert:
                vocab_name = 'vocab.txt'
            else:
                raise ValueError()
            vocab_path = os.path.join(vocab_dir, vocab_name)
            command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(output_dir, vocab_name))
            subprocess.run(command, shell=True)

    def load_data(self, data_path, flag):
        lines = read_tsv(data_path)

        data = []

        for sentence, label in lines:

            if len(sentence) > self.max_seq_length:
                continue

            word_list = None
            word_matching_position = None
            tag_list = None
            tag_matching_position = None

            type_matching = []

            type_list = None

            n_b = []

            # use lexicon
            chunk_start = 0
            chunk_end = 0

            lower_sent = [w.lower() for w in sentence]
            for i in range(len(lower_sent)):
                for length in range(min(self.max_ngram_length, len(lower_sent) - i), 0, -1):
                    ngram = tuple(lower_sent[i: i + length])
                    if ngram in self.gram2id:
                        if chunk_start <= i < chunk_end:
                            chunk_end = max(chunk_end, i + length)
                            continue
                        else:
                            if chunk_end > chunk_start:
                                n_b.append((chunk_start, chunk_end - 1))
                            chunk_start = i
                            chunk_end = i + length
            if chunk_end > chunk_start:
                n_b.append((chunk_start, chunk_end - 1))

            ngram_index = []

            # in-chunk edges, adjacent
            if self.use_in_chunk:
                for s, e in n_b:
                    if not s == e:
                        for i in range(s, e):
                            type_matching.append((i, i + 1, None))
                            type_matching.append((i + 1, i, None))
                    ngram_index.append((s, e))

            # cross-chunk edges, adjacent
            if self.use_cross_chunk:
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
                        label_ids.append(0)
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

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              dep_adjacency_matrix=adjacency_matrix))
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.long)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)

        if self.gcn is not None:
            all_dep_adjacency_matrix = torch.tensor([f.dep_adjacency_matrix for f in feature], dtype=torch.float)
            dep_adjacency_matrix = all_dep_adjacency_matrix.to(device)
        else:
            dep_adjacency_matrix = None

        return input_ids, input_mask, l_mask, label_ids, segment_ids, valid_ids, \
               dep_adjacency_matrix


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
