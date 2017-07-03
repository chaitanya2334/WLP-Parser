import pandas as pd
from tabulate import tabulate

import features_config as cfg
from corpus.Manager import Manager
from preprocessing.feature_engineering import features
from preprocessing.feature_engineering.datasets import load_windows, generate_examples, load_articles
from preprocessing.utils import quicksave


def generate_features(articles):
    print("Creating features...")
    features_list = features.create_features()

    # Initialize the window generator
    # each window has a fixed maximum size of tokens

    print("Loading windows with features {0} ...".format([type(feature).__name__ for feature in features_list]))

    windows = load_windows(articles, nb_skip=0,
                           features=features_list, only_labeled_windows=True)

    # Add chains of features (each list of lists of strings)
    # and chains of labels (each list of strings)
    # to the trainer.
    # This may take a long while, especially because of the lengthy POS tagging.
    # POS tags and LDA results are cached, so the second run through this part will be significantly
    # faster.
    examples = generate_examples(windows, nb_append=sum([article.nsents for article in articles if article.status]),
                                 nb_skip=cfg.COUNT_WINDOWS_TEST, verbose=True)

    feat_seq_list = []
    y_labels = []
    x_words = []
    sent_counts = []

    for i, (features_dicts, words, labels) in enumerate(examples):
        print(i)
        print("----------------------------------")
        df = pd.DataFrame(features_dicts)
        df = df.fillna('#')
        df = pd.get_dummies(df)
        feat_seq_list.append(df.as_matrix())
        if i == 10:
            break

    for feat_seq in feat_seq_list:
        print(tabulate(feat_seq, headers='keys', tablefmt='psql'))

    return feat_seq_list

if __name__ == '__main__':
    corpus = Manager()

    articles = corpus.load_protofiles(cfg.ARTICLES_FOLDERPATH)
    res = generate_features(articles)
    quicksave(res, cfg.FEATURES_PICKLE)
