import pickle
from itertools import chain

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder

import config as cfg
from corpus.WLPDataset import WLPDataset
from postprocessing.evaluator import Evaluator


def dataset_prep(loadfile=None, savefile=None):
    if loadfile:
        print("Loading corpus ...")
        corpus = pickle.load(open(loadfile, "rb"))
    else:
        print("Loading Data ...")
        corpus = WLPDataset(gen_rel_feat=True, prep_emb=False)
        # corpus.gen_data(cfg.PER)

        if savefile:
            print("Saving corpus and embedding matrix ...")
            pickle.dump(corpus, open(savefile, "wb"))

    return corpus


def to_idx(dataset, links):
    rel_label_idx = dataset.rel_label_idx

    return [rel_label_idx[link.label] if link.label in rel_label_idx else
            rel_label_idx[cfg.NEG_REL_LABEL] for link in links]


def extract_data(start, end, dataset, feat):
    x = dataset.get_feature_vectors(feat)
    print("no of rows in dataframe:", len(x))
    print("total no of links in protocols:")
    print(sum([len(p.relations) for p in dataset.protocols]))
    # count all the links uptil the "start" th protocol
    l_start = sum([len(p.relations) for p in dataset.protocols[:start]])
    # count all the links uptil the "end" th protocol
    l_end = sum([len(p.relations) for p in dataset.protocols[:end]])
    x = x[l_start:l_end]
    print("extract_data")
    print(x.A.shape, l_end, l_start, l_end - l_start)
    links = list(chain.from_iterable([p.relations for p in dataset.protocols]))
    y = to_idx(dataset, links[l_start:l_end])

    return x, y


def single_run(dataset, ntrain, ndev, ntest, feat):
    x_train, y_train = extract_data(0, ntrain, dataset, feat)

    x_test, y_test = extract_data(ndev, ntest, dataset, feat)

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=8)

    model.fit(x_train, y_train)

    pred = model.predict(x_test)

    print(feat)
    print(classification_report(y_test, pred, target_names=cfg.RELATIONS))
    print("Macro", precision_recall_fscore_support(y_test, pred, average='macro', labels=range(len(cfg.RELATIONS))))
    print("Micro", precision_recall_fscore_support(y_test, pred, average='micro', labels=range(len(cfg.RELATIONS))))


def main():
    dataset = dataset_prep(savefile=cfg.DB_MAXENT_WITH_PARSETREES)
    total = len(dataset.protocols)

    ntrain = int(total * .60)
    ndev = int(total * .80)
    ntest = total

    word_features = ['wm1', 'hm1', 'wbnull', 'wbf', 'wbl', 'wbo', 'bm1f', 'bm1l', 'am2f', 'am2l']
    ent_features = ['et12']
    overlap_features = ['#mb', '#wb']
    chunk_features = ['cphbnull', 'cphbfl', 'cphbf', 'cphbl', 'cphbo', 'cphbm1f', 'cphbm1l', 'cpham2f', 'cpham2l']
    dep_features = ['et1dw1', 'et2dw2', 'h1dw1', 'h2dw2', 'et12SameNP', 'et12SamePP', 'et12SameVP']
    parse_features = ['ptp']

    ablation = [
        ent_features + overlap_features + chunk_features + dep_features,  # FULL
        word_features + overlap_features + chunk_features + dep_features,  # -Ent
        word_features + ent_features + chunk_features + dep_features,  # -overlap
        word_features + ent_features + overlap_features + dep_features,  # -chunk
        word_features + ent_features + overlap_features + chunk_features,  # -dep
    ]

    addition = [
                #word_features,
                #word_features + ent_features,
                #word_features + ent_features + overlap_features,
                #word_features + ent_features + overlap_features + chunk_features,
                #word_features + ent_features + overlap_features + chunk_features + dep_features,
                word_features + ent_features + overlap_features + chunk_features + dep_features + parse_features,
                ]
    for feat in addition:
        single_run(dataset, ntrain, ndev, ntest, feat)

    for feat in ablation:
        single_run(dataset, ntrain, ndev, ntest, feat)


if __name__ == '__main__':
    main()
