import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support

import config as cfg
from corpus.WLPDataset import WLPDataset


def single_run(x_train, y_train, x_test, y_test):
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=8)

    model.fit(x_train, y_train)

    pred = model.predict(x_test)

    print(classification_report(y_test, pred, target_names=cfg.RELATIONS, labels=range(len(cfg.RELATIONS))))
    print("Macro", precision_recall_fscore_support(y_test, pred, average='macro', labels=range(len(cfg.RELATIONS))))
    print("Micro", precision_recall_fscore_support(y_test, pred, average='micro', labels=range(len(cfg.RELATIONS))))


def main():
    train = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=cfg.TRAIN_ARTICLES_PATH)
    dev = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=cfg.DEV_ARTICLES_PATH)
    test = WLPDataset(gen_rel_feat=True, prep_emb=False, dir_path=cfg.TEST_ARTICLES_PATH)

    total = len(train.protocols) + len(dev.protocols) + len(test.protocols)
    train_df, y_train = train.extract_rel_data()
    test_df, y_test = test.extract_rel_data()
    pickle.dump((train_df, test_df, y_train, y_test), open("train_df.p", 'wb'))

    word_features = ['wm1', 'wbnull', 'wbf', 'wbl', 'wbo', 'bm1f', 'bm1l', 'am2f', 'am2l']
    ent_features = ['et12']
    overlap_features = ['#mb', '#wb']
    chunk_features = ['cphbnull', 'cphbfl', 'cphbf', 'cphbl', 'cphbo', 'cphbm1f', 'cphbm1l', 'cpham2f', 'cpham2l']
    dep_features = ['et1dw1', 'et2dw2', 'h1dw1', 'h2dw2', 'et12SameNP', 'et12SamePP', 'et12SameVP']

    addition = [
        word_features,
        word_features + ent_features,
        word_features + ent_features + overlap_features,
        word_features + ent_features + overlap_features + chunk_features,
        word_features + ent_features + overlap_features + chunk_features + dep_features,
    ]
    for feat in addition:
        print(feat)
        x_train = train.features.tranform(train_df, feat)
        x_test = train.features.tranform(test_df, feat)
        single_run(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
