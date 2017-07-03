import glob
import os
import random
from collections import namedtuple
import pickle

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

Dataset = namedtuple('Dataset', ['X', 'C', 'Y', 'P'])


class Manager(object):
    def __init__(self, word_index=None, char_index=None, load_file=None, save_file=None):

        self.load_file = load_file
        self.save_file = save_file
        self.word_index = word_index
        self.char_index = char_index
        self.train = [Dataset(None, None, None, None)]
        self.dev = [Dataset(None, None, None, None)]
        self.test = [Dataset(None, None, None, None)]

    def load_filenames(self, dir_path):
        filenames = self.__from_dir(dir_path, extension="ann")
        return filenames

    def load_protofiles(self, dir_path=None, filenames=None):
        if dir_path is None and filenames is None:
            raise ValueError("Both dir path and filenames are None")

        if dir_path and filenames is None:
            filenames = self.__from_dir(dir_path, extension="ann")
            print(len(filenames))

        protofiles = [ProtoFile(filename) for filename in filenames]

        return protofiles

    def __gen_data(self, per_train, per_dev, per_test, replace_digit, to_filter):

        articles = self.load_protofiles(cfg.ARTICLES_FOLDERPATH)

        # get list of list of words
        sents, labels, pno = self.__load_sents_and_labels(articles, with_bio=True)

        cfg.ver_print("sents", sents)
        cfg.ver_print("labels", labels)

        print(len(sents))
        # filter, such that only sentences that have atleast one action-verb will be taken into consideration
        if to_filter:
            sents, labels, pno = self.__filter(sents, labels, pno)

        print(len(sents))
        if replace_digit:
            sents = self.replace_num(sents)

        cfg.ver_print("char_index", self.char_index)
        char_idx_seq = [
            self.__to_idx_seq([[cfg.SENT_START]], start=cfg.WORD_START, end=cfg.WORD_END, index=self.char_index) +
            self.__to_idx_seq([list(word) for word in sent], start=cfg.WORD_START, end=cfg.WORD_END,
                              index=self.char_index) +
            self.__to_idx_seq([[cfg.SENT_END]], start=cfg.WORD_START, end=cfg.WORD_END, index=self.char_index)
            for sent in sents]

        cfg.ver_print("char idx seq", char_idx_seq)
        # convert list of list of words to list of list of word_indices

        cfg.ver_print("word_index", self.word_index)
        sent_idx_seq = self.__to_idx_seq(sents, start=cfg.SENT_START, end=cfg.SENT_END, index=self.word_index)
        cfg.ver_print("sent", sents)
        cfg.ver_print("sent idx seq", sent_idx_seq)

        labels = self.__to_categorical(labels, bio=True)

        labels = np.array(labels)

        np.set_printoptions(precision=3, threshold=np.nan)

        total = len(sent_idx_seq)
        print(total)

        ntrain = int((per_train * total) / 100.0)
        ndev = int((per_dev * total) / 100.0)
        ntest = total - ntrain - ndev

        print(ntrain, ndev, ntest)

        x_train, x_dev, x_test = self.split(sent_idx_seq, [per_train, per_dev, per_test])
        c_train, c_dev, c_test = self.split(char_idx_seq, [per_train, per_dev, per_test])
        y_train, y_dev, y_test = self.split(labels, [per_train, per_dev, per_test])
        p_train, p_dev, p_test = self.split(pno, [per_train, per_dev, per_test])

        assert len(x_train) + len(x_dev) + len(x_test) == total

        print("nTrain={0}, nDev={1}, nTest={2}".format(len(x_train), len(x_dev), len(x_test)))

        train = [Dataset(x, c, y, p) for x, c, y, p in zip(x_train, c_train, y_train, p_train)]
        dev = [Dataset(x, c, y, p) for x, c, y, p in zip(x_dev, c_dev, y_dev, p_dev)]
        test = [Dataset(x, c, y, p) for x, c, y, p in zip(x_test, c_test, y_test, p_test)]

        return train, dev, test

    @staticmethod
    def split(seq, per):
        assert sum(per) == 100
        prv = 0
        size = len(seq)
        res = ()
        cum_percentage = 0
        for p in per:
            cum_percentage += p
            nxt = int((cum_percentage/100) * size)
            res = res + (seq[prv:nxt],)
            prv = nxt

        return res

    def to_words(self, idx_seq):
        id2word = {v: k for k, v in self.word_index.items()}
        return [id2word[idx] for idx in idx_seq]

    def gen_data(self, per_train, per_dev, per_test, replace_digit=True, to_filter=True):

        if self.load_file:
            self.train, self.dev, self.test, self.word_index, self.char_index = pickle.load(open(self.load_file, "rb"))
        else:
            self.train, self.dev, self.test = self.__gen_data(per_train, per_dev, per_test,
                                                              replace_digit, to_filter)

        if self.save_file:
            pickle.dump((self.train, self.dev, self.test, self.word_index, self.char_index), open(self.save_file, "wb"))

    @staticmethod
    def replace_num(sents):
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
    def __filter(sents, labels, pno):
        s = []
        l = []
        p = []
        for sent, label, single_p in zip(sents, labels, pno):
            # here sent is a sentence of word sequences, and label is a sequence of labels for a sentence.

            # check if any of the labels in this sentence have POSITIVE_LABEL in them, if they do, then consider that
            # sentence, else discard that sentence.
            if b_any(cfg.POSITIVE_LABEL in x for x in label):
                s.append(sent)
                l.append(label)
                p.append(single_p)

        return s, l, p

    @staticmethod
    def __to_idx_seq(list2d, start, end, index):
        idx_seq = []
        for row in list2d:
            row_idx_seq = [index[start]]
            for item in row:
                row_idx_seq.append(index[item.lower()])
            row_idx_seq.append(index[end])
            idx_seq.append(row_idx_seq)

        return idx_seq

    @staticmethod
    def __to_categorical(labels, bio=False):
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
    def __load_sents_and_labels(articles, with_bio=False, shuffle_once=True):
        sents = []
        labels = []
        pno = []
        for article in articles:
            assert isinstance(article, ProtoFile)
            if article.status:
                words, tags = article.extract_data_per_sent(with_bio)
                sents.extend(words)
                labels.extend(tags)
                pno.extend([article.basename]*len(words))
                assert len(pno) == len(sents)
        if shuffle_once:
            samples = list(zip(sents, labels, pno))
            cfg.ver_print("samples before shuffle", samples)
            random.shuffle(samples)
            cfg.ver_print("samples after shuffle", samples)
            sents, labels, pno = zip(*samples)
            cfg.ver_print("sents after unzipping", sents)
            cfg.ver_print("labels after unzipping", labels)
        return sents, labels, pno

    def load_textfiles(self, folder):
        return [TextFile(filename) for filename in self.__from_dir(folder, "txt")]

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
    def __from_dir(folder, extension):
        g = glob.iglob(folder + '/*.' + extension, recursive=True)
        return [os.path.splitext(f)[0] for f in g]
