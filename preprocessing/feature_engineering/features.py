# -*- coding: utf-8 -*-
"""
Contains:
    1. Various classes (feature generators) to convert windows (of words/tokens) to feature values.
       Each feature value is a string, e.g. "starts_with_uppercase=1", "brown_cluster=123".
    2. A method to create all feature generators.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import re
import sys
from itertools import chain, tee

import copy
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.parse.stanford import StanfordDependencyParser

# All capitalized constants come from this file
import features_config as cfg

from preprocessing.feature_engineering.pos import PosTagger
from preprocessing.feature_engineering.unigrams import Unigrams


def create_features(articles, verbose=True):
    """This method creates all feature generators.
    The feature generators will be used to convert windows of tokens to their string features.

    This function may run for a few minutes.

    Args:
        verbose: Whether to output messages.
    Returns:
        List of feature generators
        :param verbose: prints stuff if true
        :param articles: list of bit lengths that will be used as features
    """

    def print_if_verbose(msg):
        """This method prints a message only if verbose was set to True, otherwise does nothing.
        Args:
            msg: The message to print.
        """
        if verbose:
            print(msg)

    # Load the most common unigrams. These will be used as features.
    ug_all_top = Unigrams(articles, skip_first_n=cfg.UNIGRAMS_SKIP_FIRST_N,
                          max_count_words=cfg.UNIGRAMS_MAX_COUNT_WORDS)

    # Load all unigrams of person names (PER). These will be used to create the Gazetteer.
    # print_if_verbose("Loading person name unigrams...")
    # ug_names = Unigrams(cfg.UNIGRAMS_PERSON_FILEPATH)

    # Create the gazetteer. The gazetteer will contain all names from ug_names that have a higher
    # frequency among those names than among all unigrams (from ug_all).
    # print_if_verbose("Creating gazetteer...")

    # Unset ug_all and ug_names because we don't need them any more and they need quite a bit of
    # RAM.

    # Load the mapping of word to brown cluster and word to brown cluster bitchain
    # print_if_verbose("Loading brown clusters...")
    # brown = BrownClusters(cfg.BROWN_CLUSTERS_FILEPATH)

    # Load the mapping of word to word2vec cluster
    # print_if_verbose("Loading W2V clusters...")
    # w2vc = W2VClusters(cfg.W2V_CLUSTERS_FILEPATH)

    # Load the wrapper for the gensim LDA
    # print_if_verbose("Loading LDA...")
    # lda = LdaWrapper(cfg.LDA_MODEL_FILEPATH, cfg.LDA_DICTIONARY_FILEPATH,
    #                 cache_filepath=cfg.LDA_CACHE_FILEPATH)

    # Load the wrapper for the stanford POS tagger
    # print_if_verbose("Loading POS-Tagger...")
    pos = PosTagger(cfg.STANFORD_POS_JAR_FILEPATH, cfg.STANFORD_MODEL_FILEPATH,
                    cache_filepath=None)

    # create feature generators
    result = [
        # EntityTypeFeatures(),
        # NearestEntityFeatures(),
        LemmatizerFeatures(pos, ug_all_top),
        # DepGraphFeatures(),
        # DepTypeFeatures(),
        # BrownClusterFeature(brown),
        # BrownClusterBitsFeature(brown, brown_bit_series),
        BigramFeature(ug_all_top),
        UnigramFeature(ug_all_top),
        POSTagFeature(pos),

    ]

    return result


class EntityTypeFeatures(object):
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            l = token.label
            if "Action-Verb" in l:
                l = "O"
            result.append(["label=%s" % l])
        return result


class NearestEntityFeatures(object):
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass

    @staticmethod
    def get_nearest(i, window):
        tokens = window.tokens
        m = max(i, len(tokens) - i - 1)

        for w in range(m):
            if i - w >= 0:
                if tokens[i - w].label != cfg.NO_NE_LABEL and tokens[i - w].label.find('Action-Verb') == -1:
                    return tokens[i - w]
            if i + w < len(tokens):
                if tokens[i + w].label != cfg.NO_NE_LABEL and tokens[i + w].label.find('Action-Verb') == -1:
                    return tokens[i + w]

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for i, token in enumerate(window.tokens):
            res = self.get_nearest(i, window)
            if res is None:
                l = '#'
            else:
                l = res.label
            result.append(["near=%s" % l])
        return result


class DepTypeFeatures(object):
    def __init__(self):
        # self.dep_parser = dep_parser
        pass

    @staticmethod
    def get_rel(triples, word):
        rels = []
        for g, r, d in triples:
            if d[0] == word:
                rels.append(r)

        if rels:
            return rels[0]
        return "#"

    def convert_window(self, window):
        dep_graph = window.dep
        # dep_graph = self.dependency_parse(window)

        t = dep_graph.triples()
        t = list(t)
        result = []
        for token in window.tokens:
            rel = self.get_rel(t, token.word)
            result.append(["rel={0}".format(rel)])

        return result


class DepGraphFeatures(object):
    def __init__(self):
        # self.dep_parser = dep_parser
        pass

    @staticmethod
    def get_deps(triples, word):
        dep = (0, 0)
        for g, r, d in triples:
            if g[0] == word:
                dep = d
                return dep

        return dep

    @staticmethod
    def get_govs(triples, word):
        gov = (0, 0)
        for g, r, d in triples:
            if d[0] == word:
                gov = g
        return gov

    def convert_window(self, window):
        dep_graph = window.dep
        t = dep_graph.triples()
        t = list(t)
        result = []
        for token in window.tokens:
            dep = self.get_deps(t, token.word)
            gov = self.get_govs(t, token.word)

            word_list = ["dep={0}".format(dep[0]), "gov={0}".format(gov[0])]
            result.append(word_list)

        return result


class LemmatizerFeatures(object):
    # syn + lemma

    def __init__(self, pos, unigrams):
        self.pos_tagger = pos
        self.unigrams = unigrams
        self.wordnet_lemmatizer = WordNetLemmatizer()

    @staticmethod
    def is_noun(tag):
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']

    @staticmethod
    def is_verb(tag):
        return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    @staticmethod
    def is_adverb(tag):
        return tag in ['RB', 'RBR', 'RBS']

    @staticmethod
    def is_adjective(tag):
        return tag in ['JJ', 'JJR', 'JJS']

    def convert_tag(self, pos_tag):
        if self.is_adjective(pos_tag):
            return wordnet.ADJ
        elif self.is_noun(pos_tag):
            return wordnet.NOUN
        elif self.is_adverb(pos_tag):
            return wordnet.ADV
        elif self.is_verb(pos_tag):
            return wordnet.VERB
        return wordnet.NOUN

    def convert_window(self, window):
        # print("lemma ...", end=" ")
        sys.stdout.flush()
        pos_tags = window.pos
        result = []
        sys.stdout.flush()
        # catch stupid problems with stanford POS tagger and unicode characters
        if len(pos_tags) == len(window.tokens):
            # _ is the word
            for token, (_, pos_tag) in zip(window.tokens, pos_tags):
                if self.token_to_rank(token) != "#":
                    the_word = self.wordnet_lemmatizer.lemmatize(token.word, pos=self.convert_tag(pos_tag))
                    synonyms = wordnet.synsets(the_word, pos=self.convert_tag(pos_tag))
                    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
                else:
                    lemmas = ["#"]

                result.append(["lm={0}".format("".join(lemmas))])
        else:
            orig_str = "|".join([token.word for token in window.tokens])
            pos_str = "|".join([word for word, _ in pos_tags])
            print("[Info] In Protocol {0}: Stanford POS tagger got sequence of length {1}, returned "
                  "POS-sequence of length {2}. This sometimes happens with special unicode "
                  "characters. Returning empty list instead.".format(window.pno, len(window.tokens), len(pos_tags)))
            print("[Info] Original sequence was:", orig_str)
            print("[Info] Tagged sequence      :", pos_str)

            # fill with empty feature value lists (one empty list per token)
            for _ in range(len(window.tokens)):
                result.append([])
        # print("done")
        return result

    def token_to_rank(self, token):
        """Converts a token/word to its unigram rank.
        Args:
            token: The token/word to convert.
        Returns:
            Unigram rank as integer,
            or -1 if it wasn't found among the unigrams.
        """
        return self.unigrams.get_rank_of(token.word, '#')

    def stanford_pos_tag(self, window):
        """Converts a EntityWindow (list of tokens) to their POS tags.
        Args:
            window: EntityWindow object containing the token list to POS-tag.
        Returns:
            List of POS tags as strings.
        """
        # print([token.word for token in window.tokens])
        return self.pos_tagger.tag([token.word for token in window.tokens])


class StartsWithUppercaseFeature(object):
    """Generates a feature that describes, whether a given token starts with an uppercase letter."""

    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["swu=%d" % (int(token.word[:1].istitle()))])
        return result


class TokenLengthFeature(object):
    """Generates a feature that describes the character length of a token."""

    def __init__(self, max_length=30):
        """Instantiates a new object of this feature generator.
        Args:
            max_length: The max length to return in the generated features, e.g. if set to 30 you
                will never get a "l=31" result, only "l=30" for a token with length >= 30.
        """
        self.max_length = max_length

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["l=%d" % (min(len(token.word), self.max_length))])
        return result


class ContainsDigitsFeature(object):
    """Generates a feature that describes, whether a token contains any digit."""

    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexp_contains_digits = re.compile(r'[0-9]+')

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            any_digits = self.regexp_contains_digits.search(token.word) is not None
            result.append(["cD=%d" % (int(any_digits))])
        return result


class ContainsPunctuationFeature(object):
    """Generates a feature that describes, whether a token contains any punctuation."""

    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexp_contains_punctuation = re.compile(r'[\.\,\:\;\(\)\[\]\?\!]+')

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            any_punct = self.regexp_contains_punctuation.search(token.word) is not None
            result.append(["cP=%d" % (int(any_punct))])
        return result


class OnlyDigitsFeature(object):
    """Generates a feature that describes, whether a token contains only digits."""

    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexp_contains_only_digits = re.compile(r'^[0-9]+$')

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            only_digits = self.regexp_contains_only_digits.search(token.word) is not None
            result.append(["oD=%d" % (int(only_digits))])
        return result


class OnlyPunctuationFeature(object):
    """Generates a feature that describes, whether a token contains only punctuation."""

    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexp_contains_only_punctuation = re.compile(r'^[\.\,\:\;\(\)\[\]\?\!]+$')

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            only_punct = self.regexp_contains_only_punctuation.search(token.word) is not None
            result.append(["oP=%d" % (int(only_punct))])
        return result


class W2VClusterFeature(object):
    """Generates a feature that describes the word2vec cluster of the token."""

    def __init__(self, w2v_clusters):
        """Instantiates a new object of this feature generator.
        Args:
            w2v_clusters: An instance of W2VClusters as defined in w2v.py that can be queried to
                estimate the cluster of a word.
        """
        self.w2v_clusters = w2v_clusters

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["w2v=%d" % (self.token_to_cluster(token))])
        return result

    def token_to_cluster(self, token):
        """Converts a token/word to its cluster index among the word2vec clusters.
        Args:
            token: The token/word to convert.
        Returns:
            cluster index as integer,
            or -1 if it wasn't found among the w2v clusters.
        """
        return self.w2v_clusters.get_cluster_of(token.word, -1)


class BrownClusterFeature(object):
    """Generates a feature that describes the brown cluster of the token."""

    def __init__(self, brown_clusters):
        """Instantiates a new object of this feature generator.
        Args:
            brown_clusters: An instance of BrownClusters as defined in brown.py that can be queried
                to estimate the brown cluster of a word.
        """
        self.brown_clusters = brown_clusters

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["bc=%d" % (self.token_to_cluster(token))])
        return result

    def token_to_cluster(self, token):
        """Converts a token/word to its cluster index among the brown clusters.
        Args:
            token: The token/word to convert.
        Returns:
            cluster index as integer,
            or -1 if it wasn't found among the brown clusters.
        """
        return self.brown_clusters.get_cluster_of(token.word, -1)


class BrownClusterBitsFeature(object):
    """Generates a feature that contains the brown cluster bitchain of the token."""

    def __init__(self, brown_clusters, bit_series):
        """Instantiates a new object of this feature generator.
        Args:
            brown_clusters: An instance of BrownClusters as defined in brown.py that can be queried
                to estimate the brown cluster bitchain of a word.
        """
        self.bit_series = bit_series
        self.brown_clusters = brown_clusters

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["bcb{0}={1}".format(bit_length, self.token_to_bitchain(token)[0:bit_length])
                           for bit_length in self.bit_series])
        return result

    def token_to_bitchain(self, token):
        """Converts a token/word to its brown cluster bitchain among the brown clusters.
        Args:
            token: The token/word to convert.
        Returns:
            brown cluster bitchain as string,
            or "" (empty string) if it wasn't found among the brown clusters.
        """
        return self.brown_clusters.get_bitchain_of(token.word, "")


class GazetteerFeature(object):
    """Generates a feature that describes, whether a token is contained in the gazetteer."""

    def __init__(self, gazetteer, name):
        """Instantiates a new object of this feature generator.
        Args:
            gazetteer: An instance of Gazetteer as defined in gazetteer.py that can be queried
                to estimate whether a word is contained in an Gazetteer.
        """
        self.gazetteer = gazetteer
        self.name = name

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        print("gaz ...", end=" ")
        for token in window.tokens:
            result.append(["g" + self.name + "=%d" % (int(self.is_in_gazetteer(token)))])
        print("done")
        return result

    def is_in_gazetteer(self, token):
        """Returns True if the token/word appears in the gazetteer.
        Args:
            token: The token/word.
        Returns:
            True if the word appears in the gazetter, False otherwise.
        """
        return self.gazetteer.contains(token.word)


class WordPatternFeature(object):
    """Generates a feature that describes the word pattern of a feature.
    A word pattern is a rough representation of the word, examples:
        original word | word pattern
        ----------------------------
        John          | Aa+
        Washington    | Aa+
        DARPA         | A+
        2055          | 9+
    """

    def __init__(self):
        """Instantiates a new object of this feature generator."""
        # maximum length of tokens after which to simply cut off
        self.max_length = 15
        # if cut off because of maximum length, use this char at the end of the word to signal
        # the cutoff
        self.max_length_char = "~"

        self.normalization = [
            (r"[A-ZÄÖÜ]", "A"),
            (r"[a-zäöüß]", "a"),
            (r"[0-9]", "9"),
            (r"[\.\!\?\,\;]", "."),
            (r"[\(\)\[\]\{\}]", "("),
            (r"[^Aa9\.\(]", "#")
        ]

        # note: we do not map numers to 9+, e.g. years will still be 9999
        self.mappings = [
            (r"[A]{2,}", "A+"),
            (r"[a]{2,}", "a+"),
            (r"[\.]{2,}", ".+"),
            (r"[\(]{2,}", "(+"),
            (r"[#]{2,}", "#+")
        ]

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        print("pat ...", end=" ")
        for token in window.tokens:
            result.append(["wp=%s" % (self.token_to_wordpattern(token))])
        print("done")
        return result

    def token_to_wordpattern(self, token):
        """Converts a token/word to its word pattern.
        Args:
            token: The token/word to convert.
        Returns:
            The word pattern as string.
        """
        normalized = token.word
        for from_regex, to_str in self.normalization:
            normalized = re.sub(from_regex, to_str, normalized)

        wpattern = normalized
        for from_regex, to_str in self.mappings:
            wpattern = re.sub(from_regex, to_str, wpattern)

        if len(wpattern) > self.max_length:
            wpattern = wpattern[0:self.max_length] + self.max_length_char

        return wpattern


class UnigramFeature(object):
    """Generates a feature that lists a unigram word, and unigram words within a +1 and -1 context window.
    """

    def __init__(self, unigrams):
        """Instantiates a new object of this feature generator.
        Args:
            unigrams: An instance of Unigrams as defined in unigrams.py that can be queried
                to estimate the rank of a word among all unigrams.
        """
        self.unigrams = unigrams

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        # print("unigram ...", end=" ")
        left_token = None
        token = None
        right_token = None

        for i in range(len(window.tokens)):
            token = window.tokens[i]
            result.append(["ng0=%s" % (self.token_to_rank(token))])

        # print("done")
        return result

    def token_to_rank(self, token):
        """Converts a token/word to its unigram rank.
        Args:
            token: The token/word to convert.
        Returns:
            Unigram rank as integer,
            or -1 if it wasn't found among the unigrams.
        """
        return self.unigrams.get_rank_of(token.word, "#")


class BigramFeature(object):
    """Generates a feature that lists a unigram word, and unigram words within a +1 and -1 context window.
    """

    def __init__(self, unigrams):
        """Instantiates a new object of this feature generator.
        Args:
            unigrams: An instance of Unigrams as defined in unigrams.py that can be queried
                to estimate the rank of a word among all unigrams.
        """
        self.unigrams = unigrams

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        # print("bigram ...", end=" ")
        left_token = None
        token = None
        right_token = None

        for i in range(len(window.tokens)):
            word1 = self.token_to_rank(window.tokens[i])
            if i < len(window.tokens) - 1:
                word2 = self.token_to_rank(window.tokens[i + 1])
            else:
                word2 = '#'
            result.append(["bg={0}{1}".format(word1, word2)])

        # print("done")
        return result

    def token_to_rank(self, token):
        """Converts a token/word to its unigram rank.
        Args:
            token: The token/word to convert.
        Returns:
            Unigram rank as integer,
            or -1 if it wasn't found among the unigrams.
        """
        return self.unigrams.get_rank_of(token.word, "#")


class PrefixFeature(object):
    """Generates a feature that describes the prefix (the first three chars) of the word."""

    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        print("pre ...", end=" ")
        for token in window.tokens:
            prefix = re.sub(r"[^a-zA-ZäöüÄÖÜß\.\,\!\?]", "#", token.word[0:3])
            result.append(["pf=%s" % (prefix)])
        print("done")
        return result


class SuffixFeature(object):
    """Generates a feature that describes the suffix (the last three chars) of the word."""

    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        print("suff ...", end=" ")
        for token in window.tokens:
            suffix = re.sub(r"[^a-zA-ZäöüÄÖÜß\.\,\!\?]", "#", token.word[-3:])
            result.append(["sf=%s" % (suffix)])
        print("done")
        return result


class POSTagFeature(object):
    """Generates a feature that describes the Part Of Speech tag of the word."""

    def __init__(self, pos_tagger):
        """Instantiates a new object of this feature generator.
        Args:
            pos_tagger: An instance of PosTagger as defined in pos.py that can be queried
                to estimate the POS-tag of a word.
        """
        self.pos_tagger = pos_tagger
        self.size = 1

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        # print("pos ...", end=" ")
        sys.stdout.flush()
        pos_tags = window.pos
        result = []
        # print("stanford_done", end=" ")
        sys.stdout.flush()
        # catch stupid problems with stanford POS tagger and unicode characters
        if len(pos_tags) == len(window.tokens):
            # _ is the word
            for pos_tag in pos_tags:
                result.append(["pos={0}".format(pos_tag[1])])
        else:
            orig_str = "|".join([token.word for token in window.tokens])
            pos_str = "|".join([word for word, _ in pos_tags])
            print("[Info] Stanford POS tagger got sequence of length %d, returned " \
                  "POS-sequence of length %d. This sometimes happens with special unicode " \
                  "characters. Returning empty list instead." % (len(window.tokens), len(pos_tags)))
            print("[Info] Original sequence was:", orig_str)
            print("[Info] Tagged sequence      :", pos_str)

            # fill with empty feature value lists (one empty list per token)
            for _ in range(len(window.tokens)):
                result.append([])
        # print("done")
        return result

    def get_context_win(self, i, pos_tags):
        # for a given size in self.size it will generate each unigram's pos tag as a seperate entry in the feature list
        result = []
        for s in range(i - self.size, i + self.size):
            if 0 < s < len(pos_tags):
                result.append()
            elif s < 0:
                result.append(["pos"])

        return result

    def stanford_pos_tag(self, window):
        """Converts a EntityWindow (list of tokens) to their POS tags.
         Args:
             window: EntityWindow object containing the token list to POS-tag.
         Returns:
             List of POS tags as strings.
         """

        return self.pos_tagger.tag([token.word for token in window.tokens])


class LDATopicFeature(object):
    """Generates a list of features that contains one or more topics of the window around the
    word."""

    def __init__(self, lda_wrapper, window_left_size, window_right_size, prob_threshold=0.2):
        """Instantiates a new object of this feature generator.
        Args:
            lda_wrapper: An instance of LdaWrapper as defined in models/lda.py that can be queried
                to estimate the LDA topics of a window around a word.
            window_left_size: Size in words/tokens to the left of a word/token to use for the LDA.
            window_right_size: See window_left_size.
            prob_threshold: The probability threshold to use for the topics. If a topic has a
                higher porbability than this threshold, it will be added as a feature,
                e.g. "lda_15=1" if topic 15 has a probability >= 0.2.
        """
        self.lda_wrapper = lda_wrapper
        self.window_left_size = window_left_size
        self.window_right_size = window_right_size
        self.prob_threshold = prob_threshold

    def convert_window(self, window):
        """Converts a EntityWindow object into a list of lists of features, where features are strings.
        Args:
            window: The EntityWindow object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for i, token in enumerate(window.tokens):
            token_features = []
            window_start = max(0, i - self.window_left_size)
            window_end = min(len(window.tokens), i + self.window_right_size + 1)
            window_tokens = window.tokens[window_start:window_end]
            text = " ".join([token.word for token in window_tokens])
            topics = self.get_topics(text)
            for (topic_idx, prob) in topics:
                if prob > self.prob_threshold:
                    token_features.append("lda_%d=%s" % (topic_idx, "1"))
            result.append(token_features)
        return result

    def get_topics(self, text):
        """Converts a small text window (string) to its LDA topics.
        Args:
            text: The small text window to convert (as string).
        Returns:
            List of tuples of form (topic index, probability).
        """
        return self.lda_wrapper.get_topics(text)
