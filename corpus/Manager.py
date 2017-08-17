import glob
import os
import random
from collections import namedtuple

from nltk.parse.stanford import StanfordDependencyParser
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate

from tqdm import tqdm

import features_config as feat_cfg

import re

from torch.utils import data

import config as cfg
from preprocessing.feature_engineering import features
from preprocessing.feature_engineering.datasets import Window
import pandas as pd

from preprocessing.feature_engineering.pos import PosTagger

from corpus.ProtoFile import ProtoFile
import itertools
from builtins import any as b_any

# TODO fix corpus folder's structure and then improve this class
from corpus.TextFile import TextFile

Data = namedtuple('Data', ['X', 'C', 'Y', 'P', 'F'])


class CustomDataset(data.Dataset):
    def __init__(self, sents, labels, cut_list, enc, f_df, pnos, char_index, word_index):
        self.sents = sents
        self.labels = labels
        self.cut_list = cut_list
        self.enc = enc
        self.f_df = f_df
        self.pnos = pnos
        self.char_index = char_index
        self.word_index = word_index

        self.tag_idx = {'B': 0, 'I': 1, 'O': 2}

    def __getitem__(self, item):
        x = self.__gen_sent_idx_seq(self.sents[item])
        c = self.__prep_char_idx_seq(self.sents[item])
        y = self.to_categorical(self.labels[item], bio=True)

        f = self.f_df[self.cut_list[item]:self.cut_list[item + 1]]
        f = self.enc.transform(f.as_matrix()).todense()
        p = self.pnos[item]
        assert len(x) == len(f) + 2, (len(x), len(f))
        return Data(x, c, y, p, f)

    def __len__(self):
        return len(self.sents)

    def __gen_sent_idx_seq(self, sent):
        cfg.ver_print("word_index", self.word_index)
        sent_idx_seq = self.__to_idx_seq(sent, start=cfg.SENT_START, end=cfg.SENT_END, index=self.word_index)
        cfg.ver_print("sent", sent)
        cfg.ver_print("sent idx seq", sent_idx_seq)

        return sent_idx_seq

    @staticmethod
    def __to_idx_seq(list1d, start, end, index):
        row_idx_seq = [index[start]]
        for item in list1d:
            row_idx_seq.append(index[item.lower()])
        row_idx_seq.append(index[end])

        return row_idx_seq

    def __prep_char_idx_seq(self, sent):
        cfg.ver_print("char_index", self.char_index)
        char_idx_seq = [self.__to_idx_seq(list(cfg.SENT_START), start=cfg.WORD_START, end=cfg.WORD_END,
                                          index=self.char_index)] + \
                       [self.__to_idx_seq(list(word), start=cfg.WORD_START, end=cfg.WORD_END, index=self.char_index)
                        for word in sent] + \
                       [self.__to_idx_seq(list(cfg.SENT_END), start=cfg.WORD_START, end=cfg.WORD_END,
                                          index=self.char_index)]

        cfg.ver_print("char idx seq", char_idx_seq)
        return char_idx_seq

    def to_categorical(self, labels, bio=False):

        # converts a list of labels to binary representation. a label is 1 if its an action-verb, else its 0
        def split(_label):
            if _label == cfg.NEG_LABEL:
                _bio_encoding = cfg.NEG_LABEL
                _tag_name = "NoTag"
            else:
                _bio_encoding, _tag_name = _label.split("-", 1)

            return _bio_encoding, _tag_name

        ret = []
        cfg.ver_print("labels", labels)
        for label in labels:
            bio_encoding, tag_name = split(label)

            if tag_name == cfg.POSITIVE_LABEL:
                ret.append(self.tag_idx[bio_encoding])
            else:
                ret.append(self.tag_idx[cfg.NEG_LABEL])

        return ret


class Manager(object):
    def __init__(self, load_feat=False, word_index=None, char_index=None, shuffle_once=True):

        self.word_index = word_index
        self.char_index = char_index
        self.articles = None
        self.total = 0
        self.load_protofiles(cfg.ARTICLES_FOLDERPATH, shuffle_once=shuffle_once)
        self.tag_idx = {'B': 0, 'I': 1, 'O': 2}
        self.sents, self.labels, self.pnos = self.__gen_data(replace_digit=True, to_filter=cfg.FILTER_ALL_NEG)
        self.total = len(self.sents)
        self.train = None
        self.dev = None
        self.test = None
        if load_feat:
            print("Loading POS ...")
            self.pos = self.__gen_pos(self.sents)
            self.feat_list = features.create_features(self.articles)
            print("Loading windows with features {0} ...".format([type(feature).__name__ for feature in self.feat_list]))

            self.enc, self.f_df, self.cut_list = self.__gen_all_features(self.sents, self.labels, self.pnos, self.pos)


    def gen_data(self, per):
        ntrain, ndev, ntest = self.__split_dataset(per, self.total)

        self.train = CustomDataset(self.sents[0:ntrain], self.labels[0:ntrain], self.cut_list[0:ntrain + 1], self.enc,
                                   self.f_df, self.pnos[0:ntrain],
                                   self.char_index, self.word_index)
        self.dev = CustomDataset(self.sents[ntrain:ndev], self.labels[ntrain:ndev], self.cut_list[ntrain:ndev + 1],
                                 self.enc, self.f_df, self.pnos[ntrain:ndev],
                                 self.char_index, self.word_index)
        self.test = CustomDataset(self.sents[ndev:ntest], self.labels[ndev:ntest], self.cut_list[ndev:ntest + 1],
                                  self.enc,
                                  self.f_df, self.pnos[ndev:ntest],
                                  self.char_index, self.word_index)

        assert len(self.train) + len(self.dev) + len(self.test) == self.total

    def __gen_all_features(self, sents, labels, pnos, pos):
        mega_df = pd.DataFrame()
        i = 0
        cut_list = [0]
        mega_list = []
        print("Loading Dep Graphs ...")
        deps = self.__gen_dep(sents, pos)
        for x, (sent, label, pno, p) in tqdm(enumerate(zip(sents, labels, pnos, pos)), desc="Collecting features",
                                             total=len(sents)):
            feature_dicts = self.__gen_single_feature(sent, label, pno, p, deps[x])
            mega_list.extend(feature_dicts)
            # mega_df = pd.concat([mega_df, df])
            cut_list.append(i + len(sent))
            i += len(sent)
        mega_df = pd.DataFrame(mega_list)
        mega_df = mega_df.fillna(0)
        print(tabulate(mega_df, headers='keys', tablefmt='psql'))
        char_cols = mega_df.dtypes.pipe(lambda x: x[x == 'object']).index

        for c in char_cols:
            mega_df[c] = pd.factorize(mega_df[c])[0]

        print(tabulate(mega_df, headers='keys', tablefmt='psql'))
        enc = OneHotEncoder()
        enc.fit(mega_df.as_matrix())

        return enc, mega_df, cut_list

    # returns pos tags for every word in each sent in `sents`.
    @staticmethod
    def __gen_pos(sents):
        pos = PosTagger(feat_cfg.STANFORD_POS_JAR_FILEPATH, feat_cfg.STANFORD_MODEL_FILEPATH,
                        cache_filepath=None)

        return pos.tag_sents(sents)

    @staticmethod
    def __gen_dep(sents, pos=None):
        dep = StanfordDependencyParser(path_to_jar=feat_cfg.STANFORD_PARSER_JAR,
                                       path_to_models_jar=feat_cfg.STANFORD_PARSER_MODEL_JAR, java_options="-mx3000m")
        if pos:
            # adding pos data to dep parser speeds up dep generation even further
            i = [sent_dep for sent_dep in dep.tagged_parse_sents(pos)]
            return i
        else:
            return dep.parse_sents(sents)

    def __gen_single_feature(self, sent, labels, pno, pos, dep):

        window = Window(sent, labels, pno, pos, dep)
        window.apply_features(self.feat_list)
        feature_dicts = []
        for word_idx in range(len(window.tokens)):
            fvl = window.get_feature_values_list(word_idx, feat_cfg.SKIPCHAIN_LEFT, feat_cfg.SKIPCHAIN_RIGHT)
            feature_dicts.append(fvl)

        #df = pd.DataFrame(feature_dicts)

        return feature_dicts

    def load_filenames(self, dir_path):
        filenames = self.__from_dir(dir_path, extension="ann")
        return filenames

    def load_protofiles(self, dir_path=None, filenames=None, shuffle_once=False):
        if dir_path is None and filenames is None:
            raise ValueError("Both dir path and filenames are None")

        if dir_path and filenames is None:
            filenames = self.__from_dir(dir_path, extension="ann")

        self.articles = [ProtoFile(filename) for filename in filenames]

        if shuffle_once:
            random.shuffle(self.articles)

        print("loaded {0} articles".format(len(self.articles)))

    def size(self, to_filter=False):
        sents, labels, pno = self.__load_sents_and_labels(self.articles, with_bio=True)
        if to_filter:
            sents, labels, pno = self.__filter(sents, labels, pno)

        return len(sents)

    # TODO test

    def __gen_data(self, replace_digit, to_filter):

        # get list of list of words
        sents, labels, pno = self.__load_sents_and_labels(self.articles, with_bio=True)

        cfg.ver_print("sents", sents)
        cfg.ver_print("labels", labels)

        # filter, such that only sentences that have atleast one action-verb will be taken into consideration
        if to_filter:
            sents, labels, pno = self.__filter(sents, labels, pno)

        if replace_digit:
            sents = self.replace_num(sents)

        # convert list of list of words to list of list of word_indices

        return sents, labels, pno

        # features = generate_features(self.articles, nb_skip)
        # i = 0
        # for x, c, y, p, (w, f) in zip(sent_idx_seq, char_idx_seq, labels, pno, features):
        #     if i < nb_skip:
        #         pass
        #         # print(self.to_words(x), p)
        #
        #     elif to_filter and not self.does_sent_have_tags(self.to_text_label(y)):
        #         print("skipping {0}... no tags found".format(i))
        #         print(self.to_words(x), p)
        #
        #     else:
        #         print(self.to_words(x)[1:-1])
        #         print(self.__replace_num(w))
        #         print(p)
        #         assert self.to_words(x)[1:-1] == self.__to_lower(self.__replace_num(w))
        #
        #         yield Dataset(x, c, y, p, (w, f))
        #     i += 1

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

    @staticmethod
    def __split_dataset(per, size):
        assert sum(per) == 100
        res = ()
        cum_per = 0
        for p in per:
            cum_per += p
            nxt = int((cum_per / 100) * size)
            res = res + (nxt,)

        return res

    @staticmethod
    def __collate(batch):
        return batch

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
