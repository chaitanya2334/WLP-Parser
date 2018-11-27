import glob
import logging
import os
from collections import namedtuple, OrderedDict

from gensim.models import KeyedVectors, Word2Vec
from tabulate import tabulate

from preprocessing.feature_engineering.GeniaTagger import GeniaTagger
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

import features_config as feat_cfg

import re

from torch.utils import data

import config as cfg
from preprocessing.feature_engineering import features, rel_features
from preprocessing.feature_engineering.datasets import EntityWindow, RelationWindow
import pandas as pd

from corpus.ProtoFile import ProtoFile
import itertools
from builtins import any as b_any

import numpy as np

from corpus.TextFile import TextFile
from preprocessing.text_processing import gen_list2id_dict
import pprint

from utils import download

Data = namedtuple('Data', ['SENT', 'X', 'C', 'Y', 'P', 'POS'])

logger = logging.getLogger(__name__)


class CustomDataset(data.Dataset):
    def __init__(self, protocols, char_index, word_index, pos_ids, tag_idx, is_oov):
        self.protocols = protocols
        self.char_index = char_index
        self.word_index = word_index
        self.pos_index = pos_ids
        self.tag_idx = tag_idx
        self.collection = self.boil_protocols()
        self.is_oov = is_oov
        self.words = list(
            itertools.chain.from_iterable([[word for word in sent] for _, sent, _, _, _ in self.collection]))
        self.vocab = set(self.words)

    def boil_protocols(self):
        # combines all prtocol data to generate a list [(sent, label, f), ..]
        collection = []
        for p in self.protocols:
            i = 0
            pno = p.protocol_name
            for token1d in p.tokens2d:
                sent = [token.word for token in token1d]
                org_sent = [token.original for token in token1d]
                labels = [token.label for token in token1d]
                f = p.f_df[i:i + len(token1d)]
                i += len(token1d)
                collection.append((org_sent, sent, labels, f, pno))

        return collection

    def __getitem__(self, item):
        org_sent, sent, labels, f, pno = self.collection[item]
        x = self.__gen_sent_idx_seq(sent)
        c = self.__prep_char_idx_seq(sent)
        y = [self.tag_idx['<s>']] + [self.tag_idx[label] for label in labels] + [self.tag_idx['</s>']]
        f_pos = f['0:pos'].values

        # add pos tag for start and end tag
        f_pos = np.insert(f_pos, 0, self.pos_index['NULL'])
        f_pos = np.insert(f_pos, f_pos.size, self.pos_index['NULL'])
        f_pos = list(f_pos.tolist())

        assert len(x) == len(f) + 2, (len(x), len(f), pno)
        assert len(x) == len(f_pos)
        return Data(org_sent, x, c, y, pno, f_pos)

    def __len__(self):
        return len(self.collection)

    def __gen_sent_idx_seq(self, sent):
        cfg.ver_print("word_index", self.word_index)
        sent_idx_seq = self.__to_idx_seq(sent, start=cfg.SENT_START, end=cfg.SENT_END,
                                         index=self.word_index, oov=self.is_oov)
        cfg.ver_print("sent", sent)
        cfg.ver_print("sent idx seq", sent_idx_seq)

        return sent_idx_seq

    @staticmethod
    def __to_idx_seq(list1d, start, end, index, oov=None):
        row_idx_seq = [index[start]]

        for item in list1d:
            if oov and oov[index[item]] == 1:
                row_idx_seq.append(index[cfg.UNK])
            else:
                row_idx_seq.append(index[item])

        row_idx_seq.append(index[end])

        return row_idx_seq

    def __prep_char_idx_seq(self, sent):
        cfg.ver_print("char_index", self.char_index)
        char_idx_seq = [self.__to_idx_seq([cfg.SENT_START], start=cfg.WORD_START, end=cfg.WORD_END,
                                          index=self.char_index)] + \
                       [self.__to_idx_seq(list(word), start=cfg.WORD_START, end=cfg.WORD_END,
                                          index=self.char_index)
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


class Features(object):
    '''
    converts each row of dataframe (filled with strings) into a one hot vector.
    '''

    def __init__(self, rel_df):
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.rel_df = rel_df
        self.char_cols = None
        self.unique_ids = None

    @staticmethod
    def filter_by_features(df, feats):
        columns = df.columns.values
        df = df[[j for i in feats for j in columns if i == j[2:]]]
        return df

    def print_stuff(self):
        print(tabulate(self.rel_df[:10], headers='keys', tablefmt='psql'))

    def tranform(self, df, feat):
        # only the columns in df are used to generate one hot vectors
        rel_df = self.filter_by_features(self.rel_df, feat)
        self.encoder.fit(rel_df.values)
        df = self.filter_by_features(df, feat)
        return self.encoder.transform(df.values)


class WLPDataset:
    def __init__(self, prep_emb=True, gen_rel_feat=False, gen_ent_feat=False, min_wcount=1, shuffle_once=True,
                 lowercase=False, replace_digits=False, dir_path=None):

        self.lowercase = lowercase
        self.replace_digits = replace_digits
        self.word_index = dict()
        self.word_counts = OrderedDict()
        self.char_index = dict()
        self.min_wcount = min_wcount
        genia = GeniaTagger(feat_cfg.GENIA_TAGGER_FILEPATH)
        print("Using GENIA POS TAGGER")
        if dir_path is None:
            dir_path = cfg.ARTICLES_FOLDERPATH

        self.protocols = self.read_protocols(skip_files=cfg.SKIP_FILES, genia=genia, gen_features=True,
                                             dir_path=dir_path)

        # not used... TODO (for cleanup phase) use.
        # self.ent_features = Features(ent_enc, ent_df)

        self.tag_idx = self.make_bio_dict(cfg.LABELS)
        self.rel_label_idx = {k: v for v, k in enumerate(cfg.RELATIONS)}
        self.rel_label_idx[cfg.NEG_REL_LABEL] = len(self.rel_label_idx)
        # self.tokens2d, self.pnos = self.__gen_data(replace_digit=cfg.REPLACE_DIGITS)

        # self.verify_tokens(self.tokens2d)
        self.p_cnt = len(self.protocols)
        self.train = None
        self.dev = None
        self.test = None
        if prep_emb:
            self.embedding_matrix = self.prepare_embeddings()
        self.is_oov = dict()

        if gen_ent_feat:
            print("Collecting all the EntityFeatureGroup Features...")
            self.feat_list = features.create_features(self.protocols)
            print(
                "Loading windows with features {0} ...".format([type(feature).__name__ for feature in self.feat_list]))
            self.enc, self.f_df = self.__gen_all_ent_features(do_dep=False)

        if gen_rel_feat:
            relations = [p.relations for p in self.protocols]
            self.rel_df = self.get_rel_fvectors(relations)
            self.features = Features(self.rel_df)

    def get_rel_fvectors(self, relations):
        print("Collecting all the relation features ...")
        rel_feat_list = rel_features.create_features()
        rel_df = self.__gen_all_rel_features(relations, rel_feat_list)
        return rel_df

    def find_oov(self):
        is_oov = dict()
        print("Train vocab size:{0}\nDev vocab size:{1}\nTest vocab size:{2}".format(
            len(self.train.vocab), len(self.dev.vocab), len(self.test.vocab)))

        with open(cfg.OOV_FILEPATH, 'w') as f:
            f.write("Out of pre-trained Vocabulary words\n")

        with open(cfg.OOV_FILEPATH, 'a') as f:
            for word, idx in self.word_index.items():
                if word in self.train.vocab:
                    is_oov[idx] = 0
                else:
                    is_oov[idx] = 1
                    f.write('{0}\n'.format(word))

        return is_oov

    def count_oov(self, words):
        return sum([1 for word in words if self.is_oov[self.word_index[word]] == 1])

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

        # remove all words in this dictionary, that have counts less than self.min_wcount
        if self.min_wcount:
            self.word_counts = {k: v for k, v in self.word_counts.items() if v >= self.min_wcount}

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word

        word_index = dict(list(zip(sorted_voc, list(range(3, len(sorted_voc) + 1 + 2)))))

        # index 0 is reserved for unknown words
        word_index[cfg.UNK] = 0
        # later, when word_index is used, evertime there is a KeyError, the index of cfg.UNK will be used

        if support_start_stop:
            word_index['<s>'] = 1
            word_index['</s>'] = 2

        # in essence word_index is a list of words in the vocabulary, along with its id. WordFeatureGroup seen in text
        # that are not in word_index will be given cfg.UNK's id.
        return word_index

    def to_idx(self, links):
        rel_label_idx = self.rel_label_idx

        return [rel_label_idx[link.label] if link.label in rel_label_idx else
                rel_label_idx[cfg.NEG_REL_LABEL] for link in links]

    def extract_rel_data(self):
        print("total no of links in protocols:")
        print(sum([len(p.relations) for p in self.protocols]))
        relations = [p.relations for p in self.protocols]
        y = self.to_idx(list(itertools.chain.from_iterable(relations)))

        return self.rel_df, y

    def prepare_embeddings(self, load_bin=True, support_start_stop=True):
        print("Preparing Embeddings ...")
        # get all the sentences each sentence is a sequence of words (list of words)
        tokens2d = list(itertools.chain.from_iterable([p.tokens2d for p in self.protocols]))

        sents = [[token.word for token in tokens1d] for tokens1d in tokens2d]
        # train a skip gram model to generate word vectors. Vectors will be of dimension given by 'size' parameter.
        print("         Loading Word2Vec ...")
        if load_bin:
            print("                     Loading a Massive File ...")
            if not os.path.isfile(cfg.PUBMED_AND_PMC_W2V_BIN):
                url = "http://evexdb.org/pmresources/vec-space-models/PubMed-and-PMC-w2v.bin"
                dirpath = os.path.dirname(cfg.PUBMED_AND_PMC_W2V_BIN)
                print("Downloading Word2Vec resource ...")
                download(url, save_filepath=cfg.PUBMED_AND_PMC_W2V_BIN)

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
            out.writelines([item + ' ' + str(self.word_index[item]) + '\n'
                            if item in self.word_index
                            else item + ' ' + str(self.word_index[cfg.UNK]) + '\n'
                            for item in sent_iter_flat])

        embedding_matrix = np.random.uniform(low=-0.01, high=0.01, size=(len(self.word_index) + 1, cfg.EMBEDDING_DIM))
        print("         Populating Embedding Matrix ...")
        with open(cfg.OOP_FILEPATH, 'w') as f:
            f.write("Out of pre-trained Vocabulary words\n")

        for word, i in self.word_index.items():
            try:
                embedding_vector = skip_gram_model[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                # not found in pre-trained word embedding list.
                with open(cfg.OOP_FILEPATH, 'a') as f:
                    f.write('{0}\n'.format(word))
                cfg.ver_print('out of pre-trained vocab word', word)

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

    def pick_protocols(self, pnames):
        print("pnames : {}".format(pnames))
        print("pname : {}".format(self.protocols[0].protocol_name))
        return list(filter(lambda p: p.protocol_name in pnames, self.protocols))

    def gen_data(self, train_p, dev_p, test_p):

        train_protocols = self.pick_protocols(train_p)
        dev_protocols = self.pick_protocols(dev_p)
        test_protocols = self.pick_protocols(test_p)

        self.train = CustomDataset(train_protocols, self.char_index, self.word_index, self.pos_ids,
                                   self.tag_idx, self.is_oov)
        self.dev = CustomDataset(dev_protocols, self.char_index, self.word_index, self.pos_ids,
                                 self.tag_idx, self.is_oov)
        self.test = CustomDataset(test_protocols, self.char_index, self.word_index, self.pos_ids,
                                  self.tag_idx, self.is_oov)

        print("train: \n\tno. of protocols = {0} \n\tno. of sents = {1}".format(
            len(self.train.protocols), len(self.train)))
        print("dev: \n\tno. of protocols = {0} \n\tno. of sents = {1}".format(
            len(self.dev.protocols), len(self.dev)))
        print("test: \n\tno. of protocols = {0} \n\tno. of sents = {1}".format(
            len(self.test.protocols), len(self.test)))

        self.is_oov = self.find_oov()
        print("out of vocab words in dev set: {0}".format(self.count_oov(self.dev.vocab)))
        print("out of vocab words in test set: {0}".format(self.count_oov(self.test.vocab)))

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

    def __gen_all_rel_features(self, relations, rel_feat_list):
        mega_list = []

        for rlist in tqdm(relations):
            f_dicts = self.__gen_single_rel_feature(rlist, rel_feat_list)
            mega_list.extend(f_dicts)

        mega_df = pd.DataFrame(mega_list)
        mega_df = mega_df.fillna("#")
        print(list(mega_df.columns.values))
        print("mega_df shape", mega_df.shape)
        return mega_df

    @staticmethod
    def __gen_single_rel_feature(relations, rel_feat_list):
        # expect all information to be packed in each link in links
        window = RelationWindow(relations)
        window.apply_features(rel_feat_list)
        feature_dicts = []
        for link_idx in range(len(window.relations)):
            # do this only if we need to get features from left and right side of the links as well.
            fvl = window.get_feature_values_list(link_idx, feat_cfg.SKIPCHAIN_LEFT, feat_cfg.SKIPCHAIN_RIGHT)
            feature_dicts.append(fvl)

        # df = pd.DataFrame(feature_dicts)

        return feature_dicts

    def __gen_all_ent_features(self, do_dep=False):
        # updates each protocol in self.protocols with its feature set.
        i = 0
        p_cut_list = [0]
        mega_list = []

        print("Loading Dep Graphs ...")
        for p in tqdm(self.protocols, desc="Collecting features"):
            if len(p.tokens2d) != len(p.pos_tags):
                print(p.protocol_name, self.__get_missing(p.tokens2d, p.pos_tags))

            if do_dep:
                deps = p.get_deps()

            for x, (tokens1d, pos) in enumerate(zip(p.tokens2d, p.pos_tags)):
                pno = p.protocol_name
                if do_dep and deps:
                    d = deps[x]
                else:
                    d = None

                feature_dicts = self.__gen_single_feature(tokens1d, pno, pos, d)
                mega_list.extend(feature_dicts)

            p_cut_list.append(i + p.word_cnt)
            i += p.word_cnt

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

        if do_dep:

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
            for word in dep_words:
                try:
                    f_dep.append(self.word_index[word])
                except KeyError:
                    f_dep.append(self.word_index[cfg.UNK])

            self.f_dep = f_dep

        print(tabulate(mega_df[:10], headers='keys', tablefmt='psql'))
        enc = OneHotEncoder()
        enc.fit(mega_df.as_matrix())

        # redistribute the one hot features generated into each protocol
        for i, p in enumerate(self.protocols):
            p.f_df = mega_df[p_cut_list[i]:p_cut_list[i + 1]]

        return enc, mega_df

    def __gen_single_feature(self, tokens1d, pno, pos, dep=None):
        window = EntityWindow(tokens1d, pno, pos, dep)
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

    def read_protocols(self, gen_features, skip_files, genia=None, dir_path=None, filenames=None):
        if dir_path is None and filenames is None:
            raise ValueError("Both dir path and filenames are None")

        if dir_path and filenames is None:
            filenames = self.__from_dir(dir_path, extension="ann")
        if cfg.FILTER_ALL_NEG:
            print("FILTERING BAD SENTENCES")

        if skip_files:
            print(filenames)
            filenames = [filename for filename in filenames if filename not in skip_files]

        articles = [ProtoFile(filename, genia, gen_features, self.lowercase, self.replace_digits, cfg.FILTER_ALL_NEG)
                    for filename in tqdm(filenames)]

        # remove articles that are empty
        articles = [article for article in articles if article.status]

        print("\nloaded {0} articles".format(len(articles)))

        return articles

    def pos_table(self, label, pos_tags_allowed):
        counter = dict()
        c = 0
        for p in self.protocols:
            tokens = list(itertools.chain.from_iterable(p.tokens2d))  # convert 2d list of tokens into 1d
            pos_tags = list(itertools.chain.from_iterable(p.pos_tags))
            for pos_tag, token in zip(pos_tags, tokens):
                if token.label in label:
                    c += 1
                    tag = pos_tag[1] if pos_tag[1] in pos_tags_allowed else 'OTHER'
                    if tag in counter:
                        if token.word in counter[tag]:
                            counter[tag][token.word] += 1
                        else:
                            counter[tag][token.word] = 1
                    else:
                        counter[tag] = {token.word: 1}

        sorted_counter = dict()
        total_counter = dict()
        for k, v in counter.items():
            sorted_counter[k] = OrderedDict(sorted(v.items(), key=lambda kv: kv[1], reverse=True))
            total_counter[k] = sum(v.values())
        pp = pprint.PrettyPrinter(indent=4)

        pp.pprint(sorted_counter)
        pp.pprint(total_counter)
        print("total: {}".format(c))

    def ent_table(self):
        counter = dict()
        for p in self.protocols:
            for tag in p.tags:
                words = " ".join(tag.words)
                if tag.tag_name in counter:
                    if words in counter[tag.tag_name]:
                        counter[tag.tag_name][words] += 1
                    else:
                        counter[tag.tag_name][words] = 1
                else:
                    counter[tag.tag_name] = {words: 1}

        sorted_counter = dict()
        total_counter = dict()
        for k, v in counter.items():
            sorted_counter[k] = OrderedDict(sorted(v.items(), key=lambda kv: kv[1], reverse=True))
            total_counter[k] = sum(v.values())
        pp = pprint.PrettyPrinter(indent=4)

        pp.pprint(sorted_counter)
        pp.pprint(total_counter)

    def size(self, to_filter=False):
        tokens2d, pno = self.__load_sents_and_labels(self.protocols, with_bio=True)
        if to_filter:
            tokens2d, pno = self.__filter(tokens2d, pno)

        return len(tokens2d)

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

    # def split_data(self, start, stop, replace_num=True, to_filter=True):
    #    assert stop > start
    #    print(int(stop - start))
    #    return itertools.islice(self.__gen_data(replace_num, to_filter, start), int(stop - start))

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
    def load_tokenized_sents(articles):
        ret = []
        for article in articles:
            if article.status:
                ret.extend(article.sents)

        return ret

    @staticmethod
    def __from_dir(folder, extension):
        g = glob.iglob(folder + '/*/*.' + extension, recursive=True)
        return [os.path.splitext(f)[0] for f in g]
