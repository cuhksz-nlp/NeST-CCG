from collections import defaultdict
import re
import math

class FindNgrams:
    def __init__(self, min_count=0, min_pmi=0, min_freq=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.words = defaultdict(int)
        self.ngrams, self.pairs = defaultdict(int), defaultdict(int)
        self.min_freq = min_freq
        self.total = 0.

    def text_filter(self, sentence):
        cleaned_text = []
        index = 0
        for i, w in enumerate(sentence):
            if re.match(u'[^\u4e00-\u9fa50-9a-zA-Z]+', w):
                if i > index:
                    cleaned_text.append(sentence[index:i])
                index = 1 + i
        if index < len(sentence):
            cleaned_text.append(sentence[index:])

        for text in cleaned_text:
            for i in range(len(text)):
                text[i] = text[i].lower()
        return cleaned_text

    def count_ngram(self, texts, n):
        self.ngrams = defaultdict(int)
        for sentence in texts:
            for sub_sentence in self.text_filter(sentence):
                for i in range(n):
                    n_len = i + 1
                    for j in range(len(sub_sentence) - i):
                        ngram = tuple(sub_sentence[j: j+n_len])
                        self.ngrams[ngram] += 1
        self.ngrams = {i:j for i, j in self.ngrams.items() if j > self.min_count}

    def find_ngrams_pmi(self, texts, n):
        for sentence in texts:
            for sub_sentence in self.text_filter(sentence):
                self.words[sub_sentence[0]] += 1
                for i in range(len(sub_sentence)-1):
                    self.words[sub_sentence[i + 1]] += 1
                    self.pairs[(sub_sentence[i], sub_sentence[i+1])] += 1
                    self.total += 1
        self.words = {i:j for i, j in self.words.items() if j > self.min_count}
        self.pairs = {i:j for i,j in self.pairs.items() if j > self.min_count}

        self.strong_segments = set()
        for i,j in self.pairs.items():
            if i[0] in self.words and i[1] in self.words:
                mi = math.log(self.total * j / (self.words[i[0]] * self.words[i[1]]))
                if mi >= self.min_pmi:
                    self.strong_segments.add(i)

        self.ngrams = defaultdict(int)
        for sentence in texts:
            for sub_sentence in self.text_filter(sentence):
                s = [sub_sentence[0]]
                for i in range(len(sub_sentence)-1):
                    if (sub_sentence[i], sub_sentence[i+1]) in self.strong_segments:
                        s.append(sub_sentence[i+1])
                    else:
                        self.ngrams[tuple(s)] += 1
                        s = [sub_sentence[i+1]]
        self.ngrams = {i:j for i, j in self.ngrams.items() if j > 0 and len(i) <= n}
        self.renew_ngram_by_freq(texts, n)

    def renew_ngram_by_freq(self, all_sentences, n):
        new_ngram2count = {}
        for sentence in all_sentences:
            for sub_sentence in self.text_filter(sentence):
                for i in range(n):
                    n_len = i + 1
                    for j in range(len(sub_sentence) - i):
                        n_gram = tuple(sub_sentence[j: j+n_len])
                        if n_gram not in self.ngrams:
                            continue
                        if n_gram not in new_ngram2count:
                            new_ngram2count[n_gram] = 1
                        else:
                            new_ngram2count[n_gram] += 1
        self.ngrams = {gram: c for gram, c in new_ngram2count.items() if c > 0}


def read_tsv(filename):
    data = []
    sentence = []
    label = []
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                if len(sentence) > 0:
                    data.append((sentence, label, None, None, None))
                sentence = []
                label = []
                continue
            splits = line.split()
            sentence.append(splits[0])
            label.append(splits[3])
    if len(sentence) > 0:
        data.append((sentence, label, None, None, None))
    return data


def get_vocab(train_path, dev_path, max_ngram_length=5, ngram_freq_threshold=2, keep_stop_words=False):
    ngram2id = {'<PAD>': 0}
    lines = read_tsv(train_path)
    # lines += read_tsv(dev_path)
    all_sentences = [sentence[0] for sentence in lines]

    ngram_finder = FindNgrams(min_count=ngram_freq_threshold, min_freq=ngram_freq_threshold)

    ngram_finder.find_ngrams_pmi(all_sentences, max_ngram_length)

    if not keep_stop_words:
        from stop_words import stop_words
        stop_words = set(stop_words)
        ngram_dict = {}
        for ngram, c in ngram_finder.ngrams.items():
            w_list = list(ngram)
            if not all(w in stop_words for w in w_list):
                ngram_dict[ngram] = c
    else:
        ngram_dict = ngram_finder.ngrams

    ngram_count = [0 for _ in range(max_ngram_length)]
    index = 1
    for w, c in ngram_dict.items():
        ngram_count[len(list(w)) - 1] += 1
        if c >= ngram_freq_threshold and w not in ngram2id:
            ngram2id[w] = index
            index += 1

    tag2id = {'<PAD>': 0}
    tag2count = defaultdict(int)

    all_sentence_tags = [(sentence[0], sentence[1]) for sentence in lines]
    for sent, tag_list in all_sentence_tags:
        for i in range(len(sent)):
            for n in range(1, max_ngram_length+1):
                if i + n > len(sent):
                    break
                ngram = tuple(sent[i: i+n])
                tag_ngram = tuple(tag_list[i: i+n])
                if ngram in ngram2id:
                    tag2count[tag_ngram] += 1
    index = 1
    for tag, count in tag2count.items():
        if count > ngram_freq_threshold:
            tag2id[tag] = index
            index += 1

    return ngram2id, ngram_count, tag2id, tag2count, ngram_finder.strong_segments


def get_labels(train_path):
    lines = read_tsv(train_path)

    all_labels = [sentence[1] for sentence in lines]
    label2id = {'<PAD>': 0, '<UNK>': 1}
    index = 2
    for label_list in all_labels:
        for label in label_list:
            if label not in label2id:
                label2id[label] = index
                index += 1
    label2id['[CLS]'] = index
    label2id['[SEP]'] = index + 1
    assert len(label2id) == index + 2
    return label2id


def get_relation_types(train_path):
    lines = read_tsv(train_path)

    all_types = [(sentence[0], sentence[4]) for sentence in lines]
    type2count = defaultdict(int)

    for word_list, type_list in all_types:
        for word, dep_type in zip(word_list, type_list):
            joint_type = word.lower() + '_' + dep_type
            # type2count[joint_type] += 1
            type2count[dep_type] += 1

    type2id = {'<PAD>': 0, '<UNK>': 1}
    index = 2

    for t, c in type2count.items():
        if c >= 20:
            assert t not in type2id
            type2id[t] = index
            index += 1

    type2id['[CLS]'] = index
    type2id['[SEP]'] = index + 1
    assert len(type2id) == index + 2
    return type2id
