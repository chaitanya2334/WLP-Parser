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
from torch.utils import data

import config as cfg
from preprocessing.feature_engineering.generate_features import generate_features
from preprocessing.utils import generate_pf_mat
from corpus.ProtoFile import ProtoFile
import itertools
from builtins import any as b_any

# TODO fix corpus folder's structure and then improve this class
from corpus.TextFile import TextFile

Dataset = namedtuple('Dataset', ['X', 'C', 'Y', 'P', 'F'])


class Manager(object):
    def __init__(self, word_index=None, char_index=None, shuffle_once=True, load_file=None, save_file=None):

        self.load_file = load_file
        self.save_file = save_file
        self.word_index = word_index
        self.char_index = char_index
        self.articles = None
        self.total = 0
        self.load_protofiles(cfg.ARTICLES_FOLDERPATH)
        self.train = []
        self.dev = []
        self.test = []
        self.per = (100, 0, 0)
        self.tag_idx = {'B': 0, 'I': 1, 'O': 2}

    def set_per(self, per):
        self.per = per

    def load_filenames(self, dir_path):
        filenames = self.__from_dir(dir_path, extension="ann")
        return filenames

    def load_protofiles(self, dir_path=None, filenames=None, shuffle_once=False):
        if dir_path is None and filenames is None:
            raise ValueError("Both dir path and filenames are None")

        if dir_path and filenames is None:
            filenames = self.__from_dir(dir_path, extension="ann")
            print(len(filenames))

        self.articles = [ProtoFile(filename) for filename in filenames]

        if shuffle_once:
            random.shuffle(self.articles)

    def size(self, to_filter=False):
        sents, labels, pno = self.__load_sents_and_labels(self.articles, with_bio=True)
        if to_filter:
            sents, labels, pno = self.__filter(sents, labels, pno)

        return len(sents)

    def __gen_data(self, replace_digit, to_filter, nb_skip):

        # get list of list of words
        sents, labels, pno = self.__load_sents_and_labels(self.articles, with_bio=True)

        cfg.ver_print("sents", sents)
        cfg.ver_print("labels", labels)

        # filter, such that only sentences that have atleast one action-verb will be taken into consideration

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

        features = generate_features(self.articles, nb_skip)
        i = 0
        for x, c, y, p, (w, f) in zip(sent_idx_seq, char_idx_seq, labels, pno, features):
            if i < nb_skip:
                pass
                # print(self.to_words(x), p)

            elif to_filter and not self.does_sent_have_tags(self.to_text_label(y)):
                print("skipping {0}... no tags found".format(i))
                print(self.to_words(x), p)

            else:
                print(self.to_words(x)[1:-1])
                print(self.__replace_num(w))
                print(p)
                assert self.to_words(x)[1:-1] == self.__to_lower(self.__replace_num(w))

                yield Dataset(x, c, y, p, (w, f))
            i += 1

    @staticmethod
    def split(seq, per):
        assert sum(per) == 100
        prv = 0
        size = len(seq)
        res = ()
        cum_percentage = 0
        for p in per:
            cum_percentage += p
            nxt = int((cum_percentage / 100) * size)
            res = res + (seq[prv:nxt],)
            prv = nxt

        return res

    def __to_lower(self, list1d):
        return [word.lower() for word in list1d]

    def to_words(self, idx_seq):
        id2word = {v: k for k, v in self.word_index.items()}
        assert id2word[self.word_index['<s>']] == '<s>'
        return [id2word[idx] for idx in idx_seq]

    def split_data(self, start, stop, replace_num=True, to_filter=True):
        assert stop > start
        print(int(stop - start))
        return itertools.islice(self.__gen_data(replace_num, to_filter, start), int(stop - start))

    def gen_train(self, replace_digit=True, to_filter=True):
        total = self.size(to_filter)
        ntrain = int((self.per[0] / 100) * total)
        print(ntrain)
        return self.split_data(0, ntrain, replace_digit, to_filter)

    def gen_dev(self, replace_digit=True, to_filter=True):
        total = self.size(to_filter)
        ntrain = int((self.per[0] / 100) * total)
        ndev = int(ntrain + (self.per[1] / 100) * total)

        return self.split_data(ntrain + 1, ndev, replace_digit, to_filter)

    def gen_test(self, replace_digit=True, to_filter=True):
        total = self.size(to_filter)
        ntrain = int((self.per[0] / 100) * total)
        ndev = int(ntrain + (self.per[1] / 100) * total)
        ntest = int(ndev + (self.per[2] / 100) * total)

        return self.split_data(ndev + 1, ntest, replace_digit, to_filter)

    def gen_data(self, per, replace_digit=True, to_filter=True):
        total = self.size(to_filter)
        print(total)

        ntrain = int((per[0] / 100) * total)
        ndev = int(ntrain + (per[1] / 100) * total)
        ntest = int(ndev + (per[2] / 100) * total)

        assert ntest == total

        self.train = list(self.split_data(0, ntrain, replace_digit, to_filter))
        self.dev = list(self.split_data(ntrain + 1, ndev, replace_digit, to_filter))
        self.test = list(self.split_data(ndev + 1, ntest, replace_digit, to_filter))

    @staticmethod
    def __replace_num(list1d):
        new_sent = [re.sub(r'\d', '0', word) for word in list1d]
        return new_sent

    def replace_num(self, sents):
        new_sents = []
        for sent in sents:
            new_sent = self.__replace_num(sent)

            new_sents.append(new_sent)
        return new_sents

    # using a word_index dictionary, converts words to their respective index.
    # sents has to be a list of list of words.

    def to_text_label(self, idx_seq):
        id2label = {v: k for k, v in self.tag_idx.items()}
        return [id2label[idx] for idx in idx_seq]

    @staticmethod
    def does_sent_have_tags(labels):
        return b_any('B' in x or 'I' in x for x in labels)

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

    def __to_categorical(self, labels, bio=False):

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
                    sent.append(self.tag_idx[bio_encoding])
                else:
                    sent.append(self.tag_idx[cfg.NEG_LABEL])

            y_train.append(sent)
        return y_train

    @staticmethod
    def __load_sents_and_labels(articles, with_bio=False):
        sents = []
        labels = []
        pno = []
        for article in articles:
            assert isinstance(article, ProtoFile)
            if article.status:
                words, tags = article.extract_data_per_sent(with_bio)
                sents.extend(words)
                labels.extend(tags)
                pno.extend([article.basename] * len(words))
                assert len(pno) == len(sents)

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
