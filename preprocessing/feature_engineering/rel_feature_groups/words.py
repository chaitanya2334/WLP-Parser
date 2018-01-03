from corpus.ProtoFile import Relation
from preprocessing.feature_engineering.datasets import RelationWindow


# TODO test class
class WordFeatureGroup(object):
    def __init__(self):
        pass

    def convert_window(self, window):
        """Converts a RelationWindow object into a list of lists of features, where features are strings.
                Args:
                    window: The EntityWindow object (defined in datasets.py) to use.
                Returns:
                    List of lists of features.
                    One list of features for each token.
                    Each list can contain any number of features (including 0).
                    Each feature is a string.
                """
        result = []
        assert isinstance(window, RelationWindow)

        for rel in window.relations:
            assert isinstance(rel, Relation)
            result.append([self.wm1(rel),  # bag-of-words in M1
                           self.hm1(rel),  # head word of M1
                           self.wbnull(rel),  # when no word in between
                           self.wbfl(rel),  # the only word in between when only one word in between
                           self.wbf(rel),  # first word in between when at least two words in between
                           self.wbl(rel),  # last word in between when at least two words in between
                           self.wbo(rel),  # other words in between except first and last words
                           self.bm1f(rel),  # first word before M1
                           self.bm1l(rel),  # second word before M1
                           self.am2f(rel),  # first word after M2
                           self.am2l(rel),  # second word after M2
                           ])

        # print("done")
        return result

    @staticmethod
    def get_words(tokens):
        return [token.word for token in tokens]

    def wm1(self, link):
        # bag-of-words in M1
        arg1_tokens = link.get_arg1_tokens()
        words = self.get_words(arg1_tokens)
        return "wm1={0}".format("_".join(words))

    def hm1(self, link):
        # head word of M1
        arg1_tokens = link.get_arg1_tokens()
        words = self.get_words(arg1_tokens)
        return "hm1={0}".format(words[-1])

    def wm2(self, link):
        # bag - of - words in M2
        arg2_tokens = link.get_arg2_tokens()
        words = self.get_words(arg2_tokens)

        return "wm2={0}".format("_".join(words))

    def hm2(self, link):
        # words.HM2(),  # head word of M2
        arg2_tokens = link.get_arg2_tokens()
        words = self.get_words(arg2_tokens)
        return "hm2={0}".format(words[-1])

    def hm12(self, link):
        # words.HM12(),  # combination of HM1 and HM2
        arg1_tokens = link.get_arg1_tokens()
        arg2_tokens = link.get_arg2_tokens()
        words1 = self.get_words(arg1_tokens)
        words2 = self.get_words(arg2_tokens)
        return "hm12={0}".format(words1[-1] + "_" + words2[-1])

    @staticmethod
    def wbnull(link):
        wb_tokens = link.get_tokens_bet()
        return "wbnull={0}".format(bool(wb_tokens))

    def wbfl(self, link):
        wb_tokens = link.get_tokens_bet()
        words = self.get_words(wb_tokens)
        if len(words) == 1:
            return "wbfl={0}".format("_".join(words))
        else:
            return "wbfl=null"

    def wbf(self, link):
        wb_tokens = link.get_tokens_bet()
        words = self.get_words(wb_tokens)
        if len(words) > 1:
            return "wbf={0}".format(words[0])
        else:
            return "wbf=null"

    def wbl(self, link):
        wb_tokens = link.get_tokens_bet()
        words = self.get_words(wb_tokens)
        if len(words) > 1:
            return "wbl={0}".format(words[-1])
        else:
            return "wbl=null"

    def wbo(self, link):
        wb_tokens = link.get_tokens_bet()
        words = self.get_words(wb_tokens)

        if len(words) > 1:
            return "wbo={0}".format("_".join(words[1:-1]))
        else:
            return "wbo=null"

    def bm1f(self, link):
        # "word1 word2 arg1"
        # return word2
        b_tokens = link.get_b_tokens(2)
        words = self.get_words(b_tokens)
        try:
            b_words = words[-1]
        except IndexError:
            b_words = "null"
        return "bm1f={0}".format(b_words)

    def bm1l(self, link):
        # "word1 word2 arg1"
        # return word1
        b_tokens = link.get_b_tokens(2)
        words = self.get_words(b_tokens)
        try:
            b_words = words[-2]
        except IndexError:
            b_words = "null"

        return "bm1l={0}".format(b_words)

    def am2f(self, link):
        # "arg2 word1 word2"
        # return word1
        a_tokens = link.get_a_tokens(1)
        words = self.get_words(a_tokens)
        try:
            a_words = words[0]
        except IndexError:
            a_words = "null"
        return "am2f={0}".format(a_words)

    def am2l(self, link):
        a_tokens = link.get_a_tokens(2)
        words = self.get_words(a_tokens)
        try:
            a_words = words[1]
        except IndexError:
            a_words = "null"
        return "am2l={0}".format(a_words)
