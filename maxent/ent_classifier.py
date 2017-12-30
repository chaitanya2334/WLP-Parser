import sklearn
import torch
import math
from itertools import chain

import time

import pickle
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder

import config as cfg
from corpus.WLPDataset import WLPDataset
from postprocessing.evaluator import Evaluator
import numpy as np


def dataset_prep(loadfile=None, savefile=None):
    start_time = time.time()

    if loadfile:
        print("Loading corpus ...")
        corpus = pickle.load(open(loadfile, "rb"))
        corpus.gen_data(cfg.PER)
    else:
        print("Loading Data ...")
        corpus = WLPDataset(gen_feat=True)
        corpus.gen_data(cfg.PER)

        if savefile:
            print("Saving corpus and embedding matrix ...")
            pickle.dump(corpus, open(savefile, "wb"))

    end_time = time.time()
    print("Ready. Input Process time: {0}".format(end_time - start_time))

    return corpus


def to_categorical(dataset, i, bio=False):
    tag_idx = dataset.tag_idx
    tokens1d = dataset.tokens2d[i]
    return [tag_idx[token.label] if token.label in tag_idx else tag_idx[cfg.NEG_LABEL] for token in tokens1d]


def extract_data(start, end, dataset, feat):
    columns = dataset.f_df.columns.values
    df = dataset.f_df[[j for i in feat for j in columns if i in j]]
    print("Current ablation feature set:")
    print(list(df.columns.values))
    enc = OneHotEncoder()
    enc.fit(df.as_matrix())
    w = list(chain.from_iterable(dataset.tokens2d[start:end]))
    w = [token.word for token in w]
    x = df[dataset.cut_list[start]:dataset.cut_list[end]]
    x = enc.transform(x.as_matrix())
    y = list(
        chain.from_iterable([to_categorical(dataset, item, bio=True) for item in range(start, end)]))
    return x, w, y


def pos_verb_only(start, end, dataset, labels):
    df = dataset.f_df['0:pos']
    df_l = df.tolist()
    print("VB is {}".format(dataset.pos_ids['VB']))
    id2pos = {v: k for k, v in dataset.pos_ids.items()}

    df_l = [labels.index(id2pos[item]) + 1 if id2pos[item] in labels else 0 for item in df_l]
    print(df_l)
    w = list(chain.from_iterable(dataset.tokens2d[start:end]))
    w = [token.word for token in w]
    x = np.asarray(df_l[dataset.cut_list[start]:dataset.cut_list[end]])
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(x) + 1))
    x = label_binarizer.transform(x)
    y = list(
        chain.from_iterable([to_categorical(dataset, item, bio=True) for item in range(start, end)]))
    return x, w, y


def main():
    dataset = dataset_prep(loadfile=cfg.DB_MAXENT)
    dataset.tag_idx['<s>'] = len(dataset.tag_idx.keys())
    dataset.tag_idx['</s>'] = len(dataset.tag_idx.keys())
    total = len(dataset.tokens2d)
    print(dataset.tag_idx)
    print(total)
    print(len(dataset.cut_list))
    ntrain = int(total * .60)
    ndev = int(total * .80)
    ntest = total

    ablation = [
        ['pos', 'ng0', 'bg', 'rel', 'dep', 'gov', 'lm'],  # FULL
        ['ng0', 'bg', 'rel', 'dep', 'gov', 'lm'],  # -POS
        ['pos', 'rel', 'dep', 'gov', 'lm'],  # -UNIGRAM/BIGRAM
        ['pos', 'ng0', 'bg', 'lm'],  # -DEPs
        ['pos', 'ng0', 'bg', 'rel', 'dep', 'gov'],  # -LEMMA
    ]

    addition = [
        ['pos'],
        ['ng0', 'bg'],
        ['ng0', 'bg', 'lm'],
        ['pos', 'ng0', 'bg'],
        ['pos', 'ng0', 'bg', 'lm'],
        ['pos', 'ng0', 'bg', 'rel', 'dep', 'gov', 'lm']
    ]
    for feat in addition:
        x_train, w_train, y_train = extract_data(0, ntrain, dataset, feat)

        # print(list(dataset.f_df.columns.values))

        x_test, w_test, y_test = extract_data(ndev, ntest, dataset, feat)

        model = LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=8)

        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        print("ALL!")
        evaluator = Evaluator("test_all", [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=dataset.tag_idx,
                              conll_eval=True)
        evaluator.append_data(0, pred, w_test, y_test)
        evaluator.classification_report()
        print("ONLY ENTITIES!")
        evaluator = Evaluator("test_ents_only", [0, 1], skip_label=['B-Action', 'I-Action'],
                              main_label_name=cfg.POSITIVE_LABEL, label2id=dataset.tag_idx,
                              conll_eval=True)
        evaluator.append_data(0, pred, w_test, y_test)
        evaluator.classification_report()


def pos():
    dataset = dataset_prep(loadfile=cfg.DB_MAXENT)
    dataset.tag_idx['<s>'] = len(dataset.tag_idx.keys())
    dataset.tag_idx['</s>'] = len(dataset.tag_idx.keys())
    # dataset.pos_table(["B-Action", "I-Action"], ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNP', 'NNS', 'JJ'])
    dataset.ent_table()
    total = len(dataset.tokens2d)
    print(dataset.tag_idx)
    print(total)
    print(len(dataset.cut_list))
    ntrain = int(total * .60)
    ndev = int(total * .80)
    ntest = total

    x_train, w_train, y_train = pos_verb_only(0, ntrain, dataset, ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

    # print(list(dataset.f_df.columns.values))

    x_test, w_test, y_test = pos_verb_only(ndev, ntest, dataset, ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=8)

    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print("ALL!")
    evaluator = Evaluator("test_all", [0, 1], main_label_name=cfg.POSITIVE_LABEL, label2id=dataset.tag_idx,
                          conll_eval=True)
    evaluator.append_data(0, pred, w_test, y_test)
    evaluator.classification_report()
    print("ONLY ENTITIES!")
    evaluator = Evaluator("test_ents_only", [0, 1], skip_label=['B-Action', 'I-Action'],
                          main_label_name=cfg.POSITIVE_LABEL, label2id=dataset.tag_idx,
                          conll_eval=True)
    evaluator.append_data(0, pred, w_test, y_test)
    evaluator.classification_report()


if __name__ == '__main__':
    main()
    # pos()
