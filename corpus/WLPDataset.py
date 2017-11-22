import glob
import logging
import os
import random
from collections import namedtuple, OrderedDict

from gensim.models import KeyedVectors, Word2Vec
from keras.preprocessing.text import Tokenizer
from nltk import DependencyGraph
from tabulate import tabulate
import pickle

from preprocessing.feature_engineering.GeniaTagger import GeniaTagger
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

import features_config as feat_cfg

import re

from torch.utils import data

import config as cfg
from preprocessing.feature_engineering import features
from preprocessing.feature_engineering.datasets import Window
import pandas as pd

from preprocessing.feature_engineering.pos import PosTagger

from corpus.ProtoFile import ProtoFile, Token
import itertools
from builtins import any as b_any

import numpy as np

# TODO fix corpus folder's structure and then improve this class
from corpus.TextFile import TextFile
from preprocessing.text_processing import gen_list2id_dict

Data = namedtuple('Data', ['X', 'C', 'Y', 'P', 'POS', 'REL', 'DEP'])

logger = logging.getLogger(__name__)


class CustomDataset(data.Dataset):
    def __init__(self, tokens2d, cut_list, enc, f_df, pnos, char_index, word_index, pos_ids, rel_ids, dep_ids,
                 f_dep, tag_idx):
        self.tokens2d = tokens2d
        self.cut_list = cut_list
        self.enc = enc
        self.f_df = f_df
        self.pnos = pnos
        self.char_index = char_index
        self.word_index = word_index
        self.pos_index = pos_ids
        self.rel_index = rel_ids
        self.dep_index = dep_ids
        self.f_dep = f_dep

        self.tag_idx = tag_idx

    def __getitem__(self, item):
        sent = [token.word for token in self.tokens2d[item]]
        labels = [token.label for token in self.tokens2d[item]]
        x = self.__gen_sent_idx_seq(sent)
        c = self.__prep_char_idx_seq(sent)
        y = [self.tag_idx[label] for label in labels]

        f = self.f_df[self.cut_list[item]:self.cut_list[item + 1]]
        f_pos = f['0:pos'].as_matrix()
        # add pos tag for start and end tag
        f_pos = np.insert(f_pos, 0, self.pos_index['NULL'])
        f_pos = np.insert(f_pos, f_pos.size, self.pos_index['NULL'])

        f_rel = f['0:rel'].as_matrix()

        # add rel tag for start and end tag
        f_rel = np.insert(f_rel, 0, self.rel_index['NULL'])
        f_rel = np.insert(f_rel, f_rel.size, self.rel_index['NULL'])

        f_dep = self.f_dep[self.cut_list[item]:self.cut_list[item + 1]]

        f_dep.insert(0, self.word_index['<s>'])
        f_dep.insert(len(f_dep), self.word_index['</s>'])
        # f = self.enc.transform(f.as_matrix()).todense()
        p = self.pnos[item]
        assert len(x) == len(f) + 2, (len(x), len(f))
        return Data(x, c, y, p, f_pos, f_rel, f_dep)

    def __len__(self):
        return len(self.tokens2d)

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
            if item in index:
                row_idx_seq.append(index[item])
            else:
                print("bad characters: {0}".format(item))
        row_idx_seq.append(index[end])

        return row_idx_seq

    def __prep_char_idx_seq(self, sent):
        cfg.ver_print("char_index", self.char_index)
        char_idx_seq = [self.__to_idx_seq([cfg.SENT_START], start=cfg.WORD_START, end=cfg.WORD_END,
                                          index=self.char_index)] + \
                       [self.__to_idx_seq(list(word), start=cfg.WORD_START, end=cfg.WORD_END, index=self.char_index)
                        for word in sent] + \
                       [self.__to_idx_seq([cfg.SENT_END], start=cfg.WORD_START, end=cfg.WORD_END,
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


class WLPDataset(object):
    def __init__(self, gen_feat=False, shuffle_once=True):

        self.word_index = dict()
        self.word_counts = OrderedDict()
        self.char_index = dict()

        self.protocols = self.read_protocols(gen_features=True, dir_path=cfg.ARTICLES_FOLDERPATH)

        self.tag_idx = self.make_bio_dict(cfg.LABELS)
        self.tokens2d, self.pnos = self.__gen_data(replace_digit=True, to_filter=cfg.FILTER_ALL_NEG)
        # self.verify_tokens(self.tokens2d)
        self.total = len(self.tokens2d)
        self.train = None
        self.dev = None
        self.test = None
        self.embedding_matrix = self.prepare_embeddings()
        if gen_feat:
            print("Collecting all the Features...")
            self.feat_list = features.create_features(self.protocols)
            print(
                "Loading windows with features {0} ...".format([type(feature).__name__ for feature in self.feat_list]))

            self.enc, self.f_df, self.cut_list, self.f_dep = self.__gen_all_features()

    def gen_word_index(self, sents, support_start_stop):
        """Updates internal vocabulary based on a list of texts.

        Required before using `texts_to_sequences` or `texts_to_matrix`.

        # Arguments
            texts: can be a list of strings,
                or a generator of strings (for memory-efficiency)
        """

        for sent in sents:
            for w in sent:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        if support_start_stop:
            word_index['<s>'] = len(word_index) + 1
            word_index['</s>'] = len(word_index) + 1

        return word_index

    def prepare_embeddings(self, load_bin=True, support_start_stop=True):
        print("Preparing Embeddings ...")
        # get all the sentences each sentence is a sequence of words (list of words)
        sents = [[token.word for token in tokens1d] for tokens1d in self.tokens2d]
        # train a skip gram model to generate word vectors. Vectors will be of dimension given by 'size' parameter.
        print("         Loading Word2Vec ...")
        if load_bin:
            print("                     Loading a Massive File ...")
            skip_gram_model = KeyedVectors.load_word2vec_format(cfg.PUBMED_AND_PMC_W2V_BIN, binary=True)
        else:
            skip_gram_model = Word2Vec(sentences=sents, size=cfg.EMBEDDING_DIM, sg=1, window=10, min_count=1,
                                       workers=4)

        cfg.ver_print("word2vec emb size", skip_gram_model.vector_size)

        sent_iter_flat = list(itertools.chain.from_iterable(sents))

        list_of_chars = list(itertools.chain.from_iterable([list(word) for word in sent_iter_flat]))

        self.word_index = self.gen_word_index(sents, support_start_stop)

        self.char_index = gen_list2id_dict(list_of_chars, insert_words=['<w>', '</w>', '<s>', '</s>'])

        print(self.char_index)

        cfg.CHAR_VOCAB = len(self.char_index.items())

        with open('test_tokenizer.txt', 'w', encoding='utf-8') as out:
            out.writelines([item + ' ' + str(self.word_index[item]) + '\n' for item in sent_iter_flat])

        embedding_matrix = np.zeros((len(self.word_index) + 1, cfg.EMBEDDING_DIM))
        embedding_matrix = np.random.uniform(low=-0.01, high=0.01, size=(len(self.word_index) + 1, cfg.EMBEDDING_DIM))
        print("         Populating Embedding Matrix ...")
        with open(cfg.OOV_FILEPATH, 'w') as f:
            f.write("Out of Vocabulary words\n")

        for word, i in self.word_index.items():
            try:
                embedding_vector = skip_gram_model[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                # not found in vocab
                # words not found in embedding index will be all-zeros.
                with open(cfg.OOV_FILEPATH, 'a') as f:
                    f.write('{0}\n'.format(word))
                cfg.ver_print('out of vocabulary word', word)

        return embedding_matrix

    @staticmethod
    def make_bio_dict(labels):
        d = dict()
        i = 0
        for label in labels:
            for pre_tag in ['B-', 'I-']:
                d[pre_tag + label] = i
                i += 1

        d['O'] = i

        return d

    def gen_data(self, per, train_per=100, gen_feat_again=False):

        ntrain, ndev, ntest = self.__split_dataset(per, self.total)

        ntrain_cut = int((train_per * ntrain) / 100)

        print(ntrain_cut)

        self.train = CustomDataset(self.tokens2d[0:ntrain_cut], self.cut_list[0:ntrain_cut + 1],
                                   self.enc, self.f_df, self.pnos[0:ntrain_cut],
                                   self.char_index, self.word_index, self.pos_ids, self.rel_ids, self.dep_ids,
                                   self.f_dep, self.tag_idx)
        self.dev = CustomDataset(self.tokens2d[ntrain:ndev], self.cut_list[ntrain:ndev + 1],
                                 self.enc, self.f_df, self.pnos[ntrain:ndev],
                                 self.char_index, self.word_index, self.pos_ids, self.rel_ids, self.dep_ids,
                                 self.f_dep, self.tag_idx)
        self.test = CustomDataset(self.tokens2d[ndev:ntest], self.cut_list[ndev:ntest + 1],
                                  self.enc,
                                  self.f_df, self.pnos[ndev:ntest],
                                  self.char_index, self.word_index, self.pos_ids, self.rel_ids, self.dep_ids,
                                  self.f_dep, self.tag_idx)

        print("train: no. of sents = {0}".format(len(self.train)))
        print("dev: no. of sents = {0}".format(len(self.dev)))
        print("test: no. of sents = {0}".format(len(self.test)))

    @staticmethod
    def __get_missing(tokens2d, pos_tags):
        list1 = [[token.word for token in tokens1d] for tokens1d in tokens2d]
        list2 = [[word for word, tag in pos] for pos in pos_tags]
        print(len(list1), len(list2))
        differences = []

        for i, _list in enumerate(list1):
            if not _list:
                print("empty index in {0}".format(i))
            if _list not in list2:
                differences.append(_list)

        return differences

    def __gen_all_features(self):
        i = 0
        cut_list = [0]
        mega_list = []

        print("Loading Dep Graphs ...")
        for protocol in tqdm(self.protocols, desc="Collecting features"):
            if len(protocol.tokens2d) != len(protocol.pos_tags):
                print(protocol.protocol_name, self.__get_missing(protocol.tokens2d, protocol.pos_tags))
            for x, (tokens1d, pos) in enumerate(zip(protocol.tokens2d, protocol.pos_tags)):
                pno = protocol.protocol_name
                deps = protocol.get_deps()
                feature_dicts = self.__gen_single_feature(tokens1d, pno, pos, deps[x])
                mega_list.extend(feature_dicts)
                # mega_df = pd.concat([mega_df, df])
                cut_list.append(i + len(tokens1d))
                i += len(tokens1d)

        mega_df = pd.DataFrame(mega_list)
        mega_df = mega_df.fillna(0)
        print(tabulate(mega_df[:10], headers='keys', tablefmt='psql'))
        char_cols = mega_df.dtypes.pipe(lambda x: x[x == 'object']).index
        print(char_cols)
        unique_ids = dict.fromkeys(char_cols)
        for c in char_cols:
            mega_df[c], unique_ids[c] = pd.factorize(mega_df[c])

        print(unique_ids)

        pos_id_list = unique_ids['0:pos'].get_values().tolist()
        self.pos_ids = {k: v for v, k in enumerate(pos_id_list)}
        self.pos_ids['NULL'] = len(self.pos_ids)

        rel_id_list = unique_ids['0:rel'].get_values().tolist()
        self.rel_ids = {k: v for v, k in enumerate(rel_id_list)}
        self.rel_ids['NULL'] = len(self.rel_ids)

        dep_id_list = unique_ids['0:gov'].get_values().tolist()
        self.dep_ids = {k: v for v, k in enumerate(dep_id_list)}
        self.dep_ids['NULL'] = len(self.dep_ids)

        f_dep = mega_df['0:gov'].as_matrix().tolist()
        dep_words = [list(self.dep_ids.keys())[list(self.dep_ids.values()).index(word_id)] for word_id in f_dep]

        # to lowercase
        dep_words = [word.lower() for word in dep_words]

        # numbers to a single representation
        dep_words = [re.sub(r'\d', '0', word) for word in dep_words]
        f_dep = [self.word_index[word] for word in dep_words]

        print(tabulate(mega_df[:10], headers='keys', tablefmt='psql'))
        enc = OneHotEncoder()
        enc.fit(mega_df.as_matrix())

        return enc, mega_df, cut_list, f_dep

    @staticmethod
    def __gen_pos_genia(sents):
        pos_tagger = GeniaTagger(feat_cfg.GENIA_TAGGER_FILEPATH)

        res = pos_tagger.parse_through_file([" ".join(sent) for sent in sents])
        print("Done Genia Tagger")
        return res

    def __gen_single_feature(self, tokens1d, pno, pos, dep):
        window = Window(tokens1d, pno, pos, dep)
        window.apply_features(self.feat_list)
        feature_dicts = []
        for word_idx in range(len(window.tokens)):
            fvl = window.get_feature_values_list(word_idx, feat_cfg.SKIPCHAIN_LEFT, feat_cfg.SKIPCHAIN_RIGHT)
            feature_dicts.append(fvl)

        # df = pd.DataFrame(feature_dicts)

        return feature_dicts

    def load_filenames(self, dir_path):
        filenames = self.__from_dir(dir_path, extension="ann")
        return filenames

    def read_protocols(self, gen_features, dir_path=None, filenames=None):
        if dir_path is None and filenames is None:
            raise ValueError("Both dir path and filenames are None")

        if dir_path and filenames is None:
            filenames = self.__from_dir(dir_path, extension="ann")

        articles = [ProtoFile(filename, gen_features) for filename in tqdm(filenames)]

        # remove articles that are empty
        articles = [article for article in articles if article.status]

        print("\nloaded {0} articles".format(len(articles)))

        return articles

    def size(self, to_filter=False):
        tokens2d, pno = self.__load_sents_and_labels(self.protocols, with_bio=True)
        if to_filter:
            tokens2d, pno = self.__filter(tokens2d, pno)

        return len(tokens2d)

    def __gen_data(self, replace_digit, to_filter):

        # get list of list of words
        tokens2d, pno = self.__load_sents_and_labels(self.protocols, with_bio=True)

        logger.debug("labels {0}".format(tokens2d))

        # filter, such that only sentences that have atleast one action-verb will be taken into consideration
        if to_filter:
            tokens2d, pno = self.__filter(tokens2d, pno)

        if replace_digit:
            tokens2d = self.replace_num(tokens2d)

        # convert list of list of words to list of list of word_indices

        return tokens2d, pno

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
    def __replace_num(tokens1d):
        new_tokens1d = [Token(re.sub(r'\d', '0', token.word), token.label) for token in tokens1d]
        return new_tokens1d

    def replace_num(self, tokens2d):
        new_tokens2d = []
        for tokens1d in tokens2d:
            new_tokens1d = self.__replace_num(tokens1d)

            new_tokens2d.append(new_tokens1d)
        return new_tokens2d

    # using a word_index dictionary, converts words to their respective index.
    # sents has to be a list of list of words.

    def to_text_label(self, idx_seq):
        id2label = {v: k for k, v in self.tag_idx.items()}
        return [id2label[idx] for idx in idx_seq]

    @staticmethod
    def does_sent_have_tags(labels):
        return b_any('B' in x or 'I' in x for x in labels)

    @staticmethod
    def __filter(tokens2d, pno):
        new_tokens2d = []
        new_pno = []
        for tokens1d, single_p in zip(tokens2d, pno):
            # here sent is a sentence of word sequences, and label is a sequence of labels for a sentence.

            # check if any of the labels in this sentence have POSITIVE_LABEL in them, if they do, then consider that
            # sentence, else discard that sentence.

            if b_any(cfg.POSITIVE_LABEL in token.label for token in tokens1d):
                new_tokens2d.append(tokens1d)
                new_pno.append(single_p)

        return new_tokens2d, new_pno

    @staticmethod
    def __load_sents_and_labels(articles, labels_allowed=None, with_bio=False):
        sents = []
        tokens = []
        pno = []
        for article in articles:
            assert isinstance(article, ProtoFile)
            if article.status:
                sents.extend(article.sents)
                tokens2d = article.tokens2d
                tokens.extend(tokens2d)
                pno.extend([article.basename] * len(article.sents))
                assert len(pno) == len(sents)

        return tokens, pno

    def load_textfiles(self, folder):
        return [TextFile(filename) for filename in self.__from_dir(folder, "txt")]

    @staticmethod
    def load_tokenized_sents(articles, to_lowercase=True):
        ret = []
        for article in articles:
            if article.status:
                ret.extend(article.sents)

        return ret

    @staticmethod
    def __from_dir(folder, extension):
        g = glob.iglob(folder + '/*.' + extension, recursive=True)
        return [os.path.splitext(f)[0] for f in g]


if __name__ == '__main__':
    pass
    # corpus = WLPDataset(gen_feat=True)
    # dep = StanfordDependencyParser(path_to_jar=feat_cfg.STANFORD_PARSER_JAR,
    #                                path_to_models_jar=feat_cfg.STANFORD_PARSER_MODEL_JAR, java_options="-mx3000m")

    # p = [[('For', 'IN'), ('each', 'DT'), ('sample', 'NN'), (',', ','), ('combine', 'VBP'), ('the', 'DT'),
    #       ('following', 'JJ'), ('reagents', 'NNS'), ('on', 'IN'), ('ice', 'NN'), ('in', 'IN'), ('nuclease-free', 'JJ'),
    #       ('microcentrifuge', 'NN'), ('tubes', 'NNS'), (':', ':'), ('_', 'NN')]]
    # dep.tagged_parse_sents(p)
