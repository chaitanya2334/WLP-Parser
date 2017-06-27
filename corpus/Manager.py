import glob
import os
import random
from collections import namedtuple

import re
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import config as cfg
from preprocessing.utils import generate_pf_mat
from corpus.ProtoFile import ProtoFile
import itertools
from builtins import any as b_any

# TODO fix corpus folder's structure and then improve this class
from corpus.TextFile import TextFile

Dataset = namedtuple('Dataset', ['X', 'C', 'Y'])


class Manager(object):
    def __init__(self, corpus_path, common_path):

        self.corpus_path = corpus_path
        self.common_path = common_path
        self.train = [Dataset(None, None, None)]
        self.dev = [Dataset(None, None, None)]
        self.test = [Dataset(None, None, None)]

    def load_protofiles(self, end_at=5):
        filenames = []
        for i in range(end_at):
            filenames.extend(self.__from_corpus(sub_folder='iteration_' + str(i), extension='ann'))

        protofiles = [ProtoFile(filename) for filename in filenames]

        return protofiles

    def gen_data(self, per_train, per_dev, per_test, word_index, char_index, replace_digit=True, to_filter=True):

        articles = self.load_protofiles()

        # get list of list of words
        sents, labels = self.load_sents_and_labels(articles, with_bio=True)

        cfg.ver_print("sents", sents)
        cfg.ver_print("labels", labels)

        # filter, such that only sentences that have atleast one action-verb will be taken into consideration
        if to_filter:
            sents, labels = self.filter(sents, labels)

        if replace_digit:
            sents = self.replace_num(sents)

        cfg.ver_print("char_index", char_index)
        char_idx_seq = [self.to_idx_seq([[cfg.SENT_START]], start=cfg.WORD_START, end=cfg.WORD_END, index=char_index) +
                        self.to_idx_seq([list(word) for word in sent], start=cfg.WORD_START, end=cfg.WORD_END, index=char_index) +
                        self.to_idx_seq([[cfg.SENT_END]], start=cfg.WORD_START, end=cfg.WORD_END, index=char_index)
                        for sent in sents]

        cfg.ver_print("char idx seq", char_idx_seq)
        # convert list of list of words to list of list of word_indices

        cfg.ver_print("word_index", word_index)
        sent_idx_seq = self.to_idx_seq(sents, start=cfg.SENT_START, end=cfg.SENT_END, index=word_index)
        cfg.ver_print("sent", sents)
        cfg.ver_print("sent idx seq", sent_idx_seq)

        labels = self.to_categorical(labels, bio=True)

        labels = np.array(labels)

        np.set_printoptions(precision=3, threshold=np.nan)

        total = len(sent_idx_seq)
        print(total)

        ntrain = int((per_train * total) / 100.0)
        ndev = int((per_dev * total) / 100.0)
        ntest = total - ntrain - ndev

        print(ntrain, ndev, ntest)

        x_train = sent_idx_seq[:ntrain]
        c_train = char_idx_seq[:ntrain]
        y_train = labels[:ntrain]
        x_dev = sent_idx_seq[ntrain + 1:ntrain + ndev + 1]
        c_dev = char_idx_seq[ntrain + 1:ntrain + ndev + 1]
        y_dev = labels[ntrain + 1:ntrain + ndev + 1]
        x_test = sent_idx_seq[-ntest:]
        c_test = char_idx_seq[-ntest:]
        y_test = labels[-ntest:]

        assert len(x_train) + len(x_dev) + len(x_test) == total

        self.train = [Dataset(x, c, y) for x, c, y in zip(x_train, c_train, y_train)]
        self.dev = [Dataset(x, c, y) for x, c, y in zip(x_dev, c_dev, y_dev)]
        self.test = [Dataset(x, c, y) for x, c, y in zip(x_test, c_test, y_test)]

    def replace_num(self, sents):
        new_sents = []
        for sent in sents:
            new_sent = []
            for word in sent:
                word = re.sub(r'\d', '0', word)
                new_sent.append(word)

            new_sents.append(new_sent)
        return new_sents

    # using a word_index dictionary, converts words to their respective index.
    # sents has to be a list of list of words.
    @staticmethod
    def filter(sents, labels):
        s = []
        l = []
        for sent, label in zip(sents, labels):
            # here sent is a sentence of word sequences, and label is a sequence of labels for a sentence.

            # check if any of the labels in this sentence have POSITIVE_LABEL in them, if they do, then consider that
            # sentence, else discard that sentence.
            if b_any(cfg.POSITIVE_LABEL in x for x in label):
                s.append(sent)
                l.append(label)

        return s, l

    @staticmethod
    def to_idx_seq(list2d, start, end, index):
        idx_seq = []
        for row in list2d:
            row_idx_seq = [index[start]]
            for item in row:
                row_idx_seq.append(index[item.lower()])
            row_idx_seq.append(index[end])
            idx_seq.append(row_idx_seq)

        return idx_seq

    @staticmethod
    def morph_labels(labels_asarray):
        labels = labels_asarray.tolist()
        new_labels = []
        for sent in labels:
            new_sent = []
            for word in sent:
                if word == 0:
                    new_sent.append([0, 1])
                elif word == 1:
                    new_sent.append([1, 0])
                else:
                    new_sent.append([0, 0])
            new_labels.append(new_sent)

        return np.array(new_labels)

    def create_pf(self, sequences):
        pf_mat_seq = []
        for sent in sequences:
            pf_mat = generate_pf_mat(len(sent))
            pf_mat_seq.append(pf_mat)

        pf_mat_seq = np.array(pf_mat_seq)
        return pf_mat_seq

    @staticmethod
    def to_categorical(labels, bio=False):
        tag_idx = {'B': 0, 'I': 1, 'O': 2}

        # converts a list of list of labels to binary representation. a label is 1 if its an action-verb, else its 0
        def split(_label):
            if _label == cfg.NEG_LABEL:
                _bio_encoding = cfg.NEG_LABEL
                _tag_name = "NoTag"
            else:
                _bio_encoding, _tag_name = _label.split("-", 1)

            return _bio_encoding, _tag_name

        y_train = []
        cfg.ver_print("labels", labels)
        for labels_sent in labels:
            sent = []
            for label in labels_sent:
                bio_encoding, tag_name = split(label)

                if tag_name == cfg.POSITIVE_LABEL:
                    sent.append(tag_idx[bio_encoding])
                else:
                    sent.append(tag_idx[cfg.NEG_LABEL])

            y_train.append(sent)
        return y_train

    @staticmethod
    def load_sents_and_labels(articles, with_bio=False, shuffle_once=True):
        sents = []
        labels = []
        for article in articles:
            assert isinstance(article, ProtoFile)
            if article.type != article.Status.EMPTY:
                words, tags = article.extract_data_per_sent(with_bio)
                sents.extend(words)
                labels.extend(tags)
        if shuffle_once:
            samples = list(zip(sents, labels))
            print(samples)
            random.shuffle(samples)
            print(samples)
            sents, labels = zip(*samples)
        return sents, labels

    @staticmethod
    def load_labels(articles):
        labels = []
        for article in articles:
            assert isinstance(article, ProtoFile)
            if article.type != article.Status.EMPTY:
                labels.extend(article.extract_tags_per_sent())

        return labels

    def load_textfiles(self):
        return [TextFile(filename) for filename in self.__from_common('txt')]

    @staticmethod
    def load_tokenized_sents(articles, to_lowercase=True):
        ret = []
        for article in articles:
            assert isinstance(article, TextFile)
            if article.type != article.Status.EMPTY:
                for sent in article.get_tokenized_sents(to_lowercase):
                    if sent:
                        ret.append(sent)

        return ret

    @staticmethod
    def load_sents(articles):
        sents = []
        for article in articles:
            assert isinstance(article, TextFile)
            if article.type != article.Status.EMPTY:
                for sent in article.get_sents():
                    if sent:
                        sents.append(sent)

        return sents

    def __from_common(self, extension):
        g = glob.iglob(self.common_path + '/*.' + extension, recursive=True)
        return [os.path.splitext(f)[0] for f in g]

    def __from_corpus(self, sub_folder, extension):
        g = glob.iglob(self.corpus_path + '/' + sub_folder + '/*.' + extension, recursive=True)
        return [os.path.splitext(f)[0] for f in g]


if __name__ == '__main__':
    def flatten_and_collect(list2d):
        flat = list(itertools.chain.from_iterable(list2d))
        return set(flat)


    corpus = Manager(cfg.ARTICLES_FOLDERPATH, cfg.COMMON_FOLDERPATH)

    sents = corpus.load_tokenized_sents(corpus.load_textfiles())
    print(sents)
    list1d = flatten_and_collect(sents)
    with open('test_tokenizer.txt', 'w', encoding='utf-8') as out:
        out.writelines([item + '\n' for item in list1d])