import pandas as pd
from tabulate import tabulate

import features_config as cfg
from preprocessing.feature_engineering import features
from preprocessing.feature_engineering.datasets import load_windows, generate_examples


def generate_features(articles, nb_skip):
    print("Creating features...")
    features_list = features.create_features(articles)

    # Initialize the window generator
    # each window has a fixed maximum size of tokens

    print("Loading windows with features {0} ...".format([type(feature).__name__ for feature in features_list]))

    windows = load_windows(articles, nb_skip=nb_skip,
                           features=features_list, only_labeled_windows=True)

    # Add chains of features (each list of lists of strings)
    # and chains of labels (each list of strings)
    # to the trainer.
    # This may take a long while, especially because of the lengthy POS tagging.
    # POS tags and LDA results are cached, so the second run through this part will be significantly
    # faster.
    examples = generate_examples(windows, nb_append=sum([article.nsents for article in articles if article.status]),
                                 nb_skip=nb_skip, verbose=True)

    feat_seq_list = []

    for i, (features_dicts, words, labels) in enumerate(examples):
        if i < nb_skip:
            print("generate_features skipping: {0}".format(i))
            yield ([], [])

        elif not features_dicts and not words and not labels:
            yield ([], [])
        else:
            print(i)
            print("----------------------------------")
            df = pd.DataFrame(features_dicts)
            df = df.fillna('#')
            df = pd.get_dummies(df)

            # print(tabulate(df, headers='keys', tablefmt='psql'))
            print("Example:")

            yield words, df.as_matrix()


