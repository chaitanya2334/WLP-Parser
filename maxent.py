import torch
import math
from itertools import chain

import time

import pickle
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate

import config as cfg
from corpus.Manager import Manager
from postprocessing.evaluator import Evaluator
from preprocessing.text_processing import prepare_embeddings


def dataset_prep(loadfile=None, savefile=None):
    start_time = time.time()

    if loadfile:
        print("Loading corpus and Embedding Matrix ...")
        corpus, embedding_matrix = pickle.load(open(loadfile, "rb"))
        corpus.gen_data(cfg.PER)
    else:
        print("Preparing Embedding Matrix ...")
        embedding_matrix, word_index, char_index = prepare_embeddings(replace_digit=cfg.REPLACE_DIGITS)
        print("Loading Data ...")
        corpus = Manager(word_index=word_index, char_index=char_index)
        corpus.gen_data(cfg.PER)

        if savefile:
            print("Saving corpus and embedding matrix ...")
            pickle.dump((corpus, embedding_matrix), open(savefile, "wb"))

    end_time = time.time()
    print("Ready. Input Process time: {0}".format(end_time - start_time))

    return corpus, embedding_matrix

def to_categorical(dataset, labels, bio=False):

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
            ret.append(dataset.tag_idx[bio_encoding])
        else:
            ret.append(dataset.tag_idx[cfg.NEG_LABEL])

    return ret

def extract_data(start, end, dataset, feat):
    columns = dataset.f_df.columns.values
    df = dataset.f_df[[j for i in feat for j in columns if i in j]]
    print(list(df.columns.values))
    enc = OneHotEncoder()
    enc.fit(df.as_matrix())
    w = list(chain.from_iterable(dataset.sents[start:end]))
    x = df[dataset.cut_list[start]:dataset.cut_list[end]]
    x = enc.transform(x.as_matrix())
    y = list(chain.from_iterable([to_categorical(dataset, dataset.labels[item], bio=True) for item in range(start, end)]))
    return x, w, y

if __name__ == '__main__':
    dataset, emb_mat = dataset_prep(loadfile=cfg.DB_WITH_FEATURES)
    total = len(dataset.sents)
    print(total)
    ntrain = int(total*.60)
    ndev = int(total*.80)
    ntest = total

    ablation = [
        ['pos'],
        ['pos', '-1:ng0'],
        ['pos', '-1:ng0', '-1:bg'],
        ['pos', '-1:ng0', '-1:bg', 'rel'],
        ['pos', '-1:ng0', '-1:bg', 'rel', 'dep', 'gov'],
        ['pos', '-1:ng0', '-1:bg', 'rel', 'dep', 'gov', 'lm'],
    ]

    for feat in ablation:
        x_train, w_train, y_train = extract_data(0, ntrain, dataset, feat)

        print(list(dataset.f_df.columns.values))

        x_test, w_test, y_test = extract_data(ntrain, ntest, dataset, feat)

        print(list(zip(w_train, w_test)))

        model = LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=8)

        clf = LogisticRegressionCV(solver='lbfgs', multi_class='multinomial', cv=2, n_jobs=8)

        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        print(feat)
        evaluator = Evaluator("test", [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=None, conll_eval=True)
        evaluator.append_data(0, pred, w_test, y_test)
        evaluator.gen_results()
        evaluator.print_results()