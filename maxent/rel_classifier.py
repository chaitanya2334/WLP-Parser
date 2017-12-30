import pickle
from itertools import chain

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import config as cfg
from corpus.WLPDataset import WLPDataset
from postprocessing.evaluator import Evaluator


def dataset_prep(loadfile=None, savefile=None):

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

    return corpus

#TODO fix
def to_categorical(dataset, i, bio=False):
    rel_label_idx = dataset.rel_label_idx
    links = dataset.protocols.links
    return [rel_label_idx[link.label] if link.label in rel_label_idx else
            rel_label_idx[cfg.NEG_REL_LABEL] for link in links]


#TODO fix
def extract_data(start, end, dataset, feat):

    w = list(chain.from_iterable(dataset.tokens2d[start:end]))
    w = [token.word for token in w]

    x = dataset.get_feature_vectors(feat)
    x = x[dataset.cut_list[start]:dataset.cut_list[end]]

    y = list(
        chain.from_iterable([to_categorical(dataset, item, bio=True) for item in range(start, end)]))
    return x, w, y


#TODO fix
def main():
    dataset = dataset_prep(loadfile=cfg.DB_MAXENT)
    dataset.tag_idx['<s>'] = len(dataset.tag_idx.keys())
    dataset.tag_idx['</s>'] = len(dataset.tag_idx.keys())
    total = len(dataset.tokens2d)

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

#TODO after that, program each feature. Then test.