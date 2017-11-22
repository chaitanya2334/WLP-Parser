# -*- coding: utf-8 -*-
"""Functions to load data from the corpus."""
from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import re
# from unidecode import unidecode
from collections import Counter, namedtuple, OrderedDict

import features_config as cfg
from corpus.ProtoFile import ProtoFile

# All capitalized constants come from this file

Coder = namedtuple('Coder', ["path", "name"])


def split_to_chunks(of_list, chunk_size):
    """Splits a list to smaller chunks.
    Args:
        of_list: The list/iterable to split.
        chunk_size: The maximum size of each smaller part/chunk.
    Returns:
        Generator of lists (i.e. list of lists).
    """
    assert of_list is not None

    for i in range(0, len(of_list), chunk_size):
        yield of_list[i:i + chunk_size]


def load_all_protocols(folderpath):
    filenames = []

    filenames.extend([os.path.splitext(f)[0] for f in
                      glob.iglob(folderpath + '/*.txt', recursive=True)])

    articles = []
    for filename in filenames:
        with open(filename + '.txt', 'r', encoding="utf-8") as f:
            print(filename)
            text = f.readlines()
            if len(text) > 1:
                articles.append(text[1])

    return articles


def conll_writer(filetrain, filedev, filetest, train=60, dev=20, test=20):
    assert train + dev + test == 100

    articles = load_protofiles(cfg.ARTICLES_FOLDERPATH, bio_encoding=True)
    sents = []

    def bad_sent(sent):
        for w, t in sent:
            if t.tag_name == "Action-Verb":
                return False

        return True

    for article in articles:
        # filter out bad sentences from each article and extend the sents list.
        sents.extend([sent for sent in article.word_tag_per_sent if not bad_sent(sent)])

    # sents is now a list of sentences. each sentence is a list of word, tag tuples. All bad sentences (sentences which
    # do not contain an Action-Verb are considered bad sentences) are removed.

    total = sum([len(sent) for sent in sents])
    count = 0

    def writer(sent, filename):
        with open(filename, 'a', encoding='utf-8') as f:
            for w, t in sent:
                if t.tag_name != "Action-Verb":
                    tag = 'O'
                else:
                    tag = t.tag_name_bio + t.tag_name

                f.write(w + " " + tag + "\n")

            f.write("\n")

    for sent in sents:
        print(sent)
        if float(count) < (total * (float(train) / 100.0)):
            writer(sent, filetrain)
            count += len(sent)

        elif float(count) < (total * (train + dev) / 100.0):
            writer(sent, filedev)
            count += len(sent)
        else:
            writer(sent, filetest)
            count += len(sent)


def load_windows(articles, features=None, nb_skip=0, every_nth_window=0,
                 only_labeled_windows=False):
    """Loads smaller windows with a maximum size per window from a generator of articles.

    Args:
        articles: Generator of articles, as provided by load_articles().
        features: Optional features to apply to each window. (Default is None, don't apply
            any features.)
        every_nth_window: How often windows are ought to be returned, e.g. a value of 3 will
            skip two windows and return the third one. This can spread the examples over more
            (different) articles. (Default is 1, return every window.)
        only_labeled_windows: If set to True, the function will only return windows that contain
            at least one labeled token (at leas one named entity). (Default is False.)
    Returns:
        Generator of Window objects, i.e. list of Window objects.
    """
    skipped = 0
    processed_windows = 0
    for article in articles:
        # count how many labels there are in the article
        assert isinstance(article, ProtoFile)

        if article.status:
            print(article.protocol_name)
            count = article.count_labels()
            # split the tokens in the article to windows
            token_windows = article.token_sents

            pno = article.protocol_name
            # token_windows = split_to_chunks(article.tokens, window_size)
            for token_window in token_windows:
                if nb_skip > skipped:
                    skipped += 1
                    yield Window([], "0")
                    continue
                window = Window(token_window, pno)
                # generate features for all tokens in the window

                if features is not None and len(window.tokens) > 0:
                    window.apply_features(features)
                yield window
                window = None
                processed_windows += 1


def generate_examples(windows, nb_append=None, nb_skip=0, verbose=True):
    """Generates example pairs of feature lists (one per token) and labels.

    Args:
        windows: The windows to generate features and labels from, see load_windows().
        nb_append: How many windows to append max or None if unlimited. (Default is None.)
        nb_skip: How many windows to skip at the start. (Default is 0.)
        verbose: Whether to print status messages. (Default is True.)
    Returns:
        Pairs of (features, labels),
        where features is a list of lists of strings,
            e.g. [["foo=bar", "asd=fgh"], ["foo=not_bar", "yikes=True"], ...]
        and labels is a list of strings,
            e.g. ["PER", "O", "O", "LOC", ...].
    """
    skipped = 0
    added = 0
    for window in windows:
        # skip the first couple of windows, if nb_skip is > 0
        if skipped < nb_skip:
            print(window.tokens)
            assert not window.tokens  # make sure that when skipping windows generator is not generating anything.
            skipped += 1
            print("skipping {0} ...".format(skipped))
            yield ({}, [], [])

        elif not window.tokens:
            yield ([], [], [])
        else:
            # chain of labels (list of strings)
            assert window.tokens  # make sure that windows has tokens.

            labels = window.get_labels()
            words = window.get_words()

            print("Processing sentence of len {0} ...".format(len(words)))
            # chain of features (list of lists of strings)
            feature_values_lists = []
            for word_idx in range(len(window.tokens)):
                fvl = window.get_feature_values_list(word_idx,
                                                     cfg.SKIPCHAIN_LEFT, cfg.SKIPCHAIN_RIGHT)
                feature_values_lists.append(fvl)
            # yield (features, labels) pair

            print("Window's protocol: {0}".format(window.pno))
            yield (feature_values_lists, words, labels)

            # print message every nth window
            # and stop if nb_append is reached
            added += 1
            if verbose and added % 5 == 0:
                if nb_append is None:
                    print("====================================")
                    print("Generated %d examples" % (added))
                    print("====================================")
                else:
                    print("====================================")
                    print("Generated %d of max %d examples" % (added, nb_append))
                    print("====================================")
            if nb_append is not None and added == nb_append:
                break

class Window:
    """Encapsulates a small window of text/tokens."""

    def __init__(self, tokens, pno, pos, dep):
        """Initialize a new Window object.

        Args:
            tokens: The tokens/words contained in the text window, provided as list of Token
                objects.
        """
        # super(Window, self).__init__("")  # because pylint complains otherwise
        self.tokens = tokens
        self.pno = pno
        self.pos = pos
        self.dep = dep

    def get_words(self):

        return [token.word for token in self.tokens]

    def count_labels(self, add_no_ne_label=False):
        """Returns how many named entity tokens appear in the article/document.

        Args:
            add_no_ne_label: Whether to also count unlabeled words. (Default is False.)
        Returns:
            Count of all named entity tokens (integer).
        """
        return sum([count[1] for count in self.get_label_counts(add_no_ne_label=add_no_ne_label)])

    def get_label_counts(self, add_no_ne_label=False):
        """Returns the count of each label in the article/document.
        Count means here: the number of words that have the label.

        Args:
            add_no_ne_label: Whether to count how often unlabeled words appear. (Default is False.)
        Returns:
            List of tuples of the form (label as string, count as integer).
        """
        if add_no_ne_label:
            counts = Counter([token.label for token in self.tokens])
        else:
            counts = Counter([token.label for token in self.tokens if token.label != cfg.NO_NE_LABEL])
        return counts.most_common()

    def apply_features(self, features):
        """Applies a list of feature generators to the tokens of this window.
        Each feature generator will then generate a list of featue values (as strings) for each
        token. Each of these lists can be empty. The lists are saved in the tokens and can later
        on be requested multiple times without the generation overhead (which can be heavy for
        some features).

        Args:
            features: A list of feature generators from features.py .
        """
        # feature_values is a multi-dimensional list
        # 1st dimension: Feature (class)
        # 2nd dimension: token
        # 3rd dimension: values (for this token and feature, usually just one value, sometimes more,
        #                        e.g. "w2vc=975")
        features_values = [feature.convert_window(self) for feature in features]

        for token in self.tokens:
            token.feature_values = []

        # After this, each self.token.feature_values will be a simple list
        # of feature values, e.g. ["w2v=875", "bc=48", ...]
        for feature_values in features_values:
            assert isinstance(feature_values, list)
            assert len(feature_values) == len(self.tokens)
            for token_idx in range(len(self.tokens)):
                self.tokens[token_idx].feature_values.extend(feature_values[token_idx])

    @staticmethod
    def convert_list_2_dict(of_list):
        of_dict = OrderedDict()
        for item in of_list:
            w1, w2 = re.search('([^=]*)=(.*)', item).groups()
            of_dict[w1] = w2

        return of_dict

    def get_feature_values_list(self, word_index, skipchain_left, skipchain_right):
        """Generates a list of feature values (strings) for one token/word in the window.

        Args:
            word_index: The index of the word/token for which to generate the featueres.
            skipchain_left: How many words to the left will be included among the features of
                the requested word. E.g. a value of 1 could lead to a list like
                ["-1:w2vc=123", "-1:l=30", "0:w2vc=18", "0:l=4"].
            skipchain_right: Like skipchain_left, but to the right side.
        Returns:
            List of strings (list of feature values).
        """
        assert word_index >= 0
        assert word_index < len(self.tokens)

        all_feature_values = []

        start = max(0, word_index - skipchain_left)
        end = min(len(self.tokens), word_index + 1 + skipchain_right)
        for i, token in enumerate(self.tokens[start:end]):
            diff = start + i - word_index
            feature_values = ["%d:%s" % (diff, feature_value) \
                              for feature_value in token.feature_values]
            all_feature_values.extend(feature_values)

        all_feature_values = self.convert_list_2_dict(all_feature_values)
        return all_feature_values

    def get_labels(self):
        """Returns the labels of all tokens as a list.
        Returns:
            list of strings"""
        return [token.label for token in self.tokens]

    def count_action(self):
        labels = self.get_labels()
        return sum([1 for label in labels if label.find('Action') != -1])


class Token(object):
    """Encapsulates a token/word.
    Members:
        token.original: The original token, i.e. the word _and_ the label, e.g. "John/PER".
        token.word: The string content of the token, without the label.
        token.label: The label of the token.
        token.feature_values: The feature values, after they have been applied.
            (See Window.apply_features().)
    """

    def __init__(self, word, label=cfg.NO_NE_LABEL):
        """Initialize a new Token object.
        Args:
            original: The original word as found in the text document, including the label,
                e.g. "foo", "John/PER".
        """
        # self.original = original

        self.word = word
        self.label = label
        # self._word_ascii = None
        self.feature_values = None
